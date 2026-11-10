//! Per-collection commit queue: WAL group commit plus a single applier.
//!
//! [`CollectionStore::apply_txn`](super::CollectionStore::apply_txn) is split
//! into four steps so that the store write lock never covers an fsync:
//!
//! 1. **reserve** — one short `inner.write()` critical section assigns the
//!    lsn, resolves the slot for every upsert and pushes the ticket;
//! 2. **flush** — the worker drains the whole backlog into one WAL batch, so
//!    concurrent writers share a single fsync (real group commit);
//! 3. **apply** — the same worker applies the batch in lsn order under a short
//!    `inner.write()`;
//! 4. **complete** — the worker settles the ticket and wakes its caller.
//!
//! Two invariants depend on this shape:
//!
//! - **log order ≡ apply order.** The lsn, the slot reservations and the queue
//!   position are decided in the same critical section, and the worker applies
//!   strictly in lsn order. Replay walks the log in that same order, so the
//!   slot numbers a reopened store rebuilds are identical to the live ones.
//! - **apply after durable.** A batch is applied only once its fsync returned.
//!   A failed batch is rolled back by the WAL and none of its transactions is
//!   applied, so nothing exists in memory that the log could not restore.
//!
//! One worker thread per collection, created on the first transaction. It is
//! the only applier, which is what keeps the ordering invariant trivially
//! true; the apply path therefore never enqueues (asserted, because waiting
//! for its own batch would deadlock).
//!
//! Lock order: `inner → queue state`. The worker takes `inner.write()` to
//! apply and the queue state to settle a ticket, in that order; writers take
//! `inner.write()` to reserve and the queue state to enqueue, in that order.
//! Quiescing (checkpoint / compaction) takes the queue state alone and never
//! holds `inner` while waiting, so no cycle forms.

use std::collections::VecDeque;
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use super::{set_applying, set_commit_worker, CollectionStore, WalRecord};
use crate::error::{Result, VectorSearchError};

/// How long a writer waits for its own transaction before giving up.
///
/// The worker settles every ticket it takes, so reaching this timeout means
/// the applier is stuck or dead; failing the write is the only recovery that
/// leaves the collection usable for reads.
pub(crate) const APPLY_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Completion slot shared between one writer and the queue worker.
pub(crate) struct Completion {
    outcome: Mutex<Option<Result<()>>>,
    settled: Condvar,
}

impl Completion {
    fn new() -> Self {
        Self {
            outcome: Mutex::new(None),
            settled: Condvar::new(),
        }
    }

    fn settle(self: &Arc<Self>, outcome: Result<()>) {
        *self.outcome.lock() = Some(outcome);
        self.settled.notify_all();
    }

    /// Block until the worker settles this ticket, or until `timeout`.
    pub(crate) fn wait(&self, timeout: Duration) -> Result<()> {
        let mut outcome = self.outcome.lock();
        if outcome.is_none() {
            let deadline = Instant::now() + timeout;
            loop {
                if self.settled.wait_until(&mut outcome, deadline).timed_out() {
                    break;
                }
                if outcome.is_some() {
                    break;
                }
            }
        }
        match outcome.take() {
            Some(outcome) => outcome,
            None => Err(VectorSearchError::Internal(
                "commit queue apply timed out".to_string(),
            )),
        }
    }
}

/// One queued transaction: its lsn, its ops and the caller waiting on it.
struct Ticket {
    lsn: u64,
    ops: Vec<WalRecord>,
    completion: Arc<Completion>,
}

struct QueueState {
    /// Tickets waiting to be flushed and applied, in lsn order.
    queue: VecDeque<Ticket>,
    /// Every unsettled ticket, whether queued or held by the worker. Keeping
    /// the completions here lets a failure settle waiters that the worker
    /// already took out of the queue.
    waiters: Vec<Arc<Completion>>,
    /// Enqueued but not yet applied transactions.
    in_flight: usize,
    /// Set while a checkpoint / compaction critical section is draining or
    /// holding the queue: new writers wait, so no WAL append can interleave
    /// with the log truncate those sections perform.
    ///
    /// Counted, never a flag: an explicit `compact_collection` call can race
    /// the maintenance worker's checkpoint for the same collection, and the
    /// first of the two finishing must not open the gate while the other is
    /// still inside its log-truncate window.
    quiescing: usize,
    /// Writers that passed the gate but have not enqueued yet. Quiescing
    /// waits for this to reach zero as well as for the queue to drain: a
    /// writer already inside the store write lock must never block on the
    /// gate, or a maintenance pass waiting for that lock would deadlock
    /// against it.
    writers: usize,
    /// Set after the worker died or an apply failed: writes fail fast.
    poisoned: bool,
    /// Worker thread, spawned by the first transaction.
    worker: Option<JoinHandle<()>>,
    /// Set when the collection is dropped: the worker exits.
    shutdown: bool,
    /// Tickets ever enqueued (monotonic). Test-only observation hook: it lets
    /// a test wait for its writers to have queued without depending on how
    /// far the worker got.
    #[cfg(test)]
    enqueued_total: u64,
    /// Test-only: park the worker before it takes its next batch, so a test
    /// can fill the queue deterministically.
    #[cfg(test)]
    hold: bool,
    /// Test-only: true while the worker is parked on `hold`.
    #[cfg(test)]
    parked: bool,
    /// Test-only: make the worker panic before flushing its next batch.
    #[cfg(test)]
    panic_before_flush: bool,
}

/// Guard over a maintenance critical section (checkpoint / compaction).
///
/// Dropping it — including while unwinding a panic — releases one gate
/// registration, so a failed maintenance pass cannot leave writers blocked
/// forever.
pub(crate) struct QuiesceGuard<'a>(&'a CommitQueue);

impl<'a> QuiesceGuard<'a> {
    /// Take one registration on the gate: writers stop entering from here.
    fn register(queue: &'a CommitQueue) -> Self {
        {
            let mut state = queue.state.lock();
            state.quiescing += 1;
        }
        Self(queue)
    }

    /// Block until no transaction is queued and no writer is on its way in.
    fn wait_drained(&self) {
        let mut state = self.0.state.lock();
        while (state.in_flight > 0 || state.writers > 0) && !state.poisoned {
            self.0.drained.wait(&mut state);
        }
    }
}

impl Drop for QuiesceGuard<'_> {
    fn drop(&mut self) {
        {
            let mut state = self.0.state.lock();
            state.quiescing = state.quiescing.saturating_sub(1);
        }
        self.0.work.notify_all();
    }
}

/// Guard for the gap between passing the queue gate and enqueueing.
///
/// Held only for the store's reserve critical section: it lets quiescing see
/// that a writer is on its way in, without that writer ever blocking on the
/// gate while it holds the store write lock.
pub(crate) struct WritePermit<'a>(&'a CommitQueue);

impl Drop for WritePermit<'_> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock();
        state.writers -= 1;
        let drained = state.writers == 0 && state.in_flight == 0;
        drop(state);
        if drained {
            self.0.drained.notify_all();
        }
    }
}

/// Commit queue of one collection: group commit in, ordered apply out.
pub(crate) struct CommitQueue {
    state: Mutex<QueueState>,
    /// Signalled when work arrives, the gate opens or shutdown starts.
    work: Condvar,
    /// Signalled when the last in-flight transaction is applied.
    drained: Condvar,
}

impl CommitQueue {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(QueueState {
                queue: VecDeque::new(),
                waiters: Vec::new(),
                in_flight: 0,
                quiescing: 0,
                writers: 0,
                poisoned: false,
                worker: None,
                shutdown: false,
                #[cfg(test)]
                enqueued_total: 0,
                #[cfg(test)]
                hold: false,
                #[cfg(test)]
                parked: false,
                #[cfg(test)]
                panic_before_flush: false,
            }),
            work: Condvar::new(),
            drained: Condvar::new(),
        }
    }

    /// Queue one transaction and return its completion slot.
    ///
    /// The caller must hold the store write lock: the lsn and the queue
    /// position have to be decided together with the slot reservations, or
    /// the log order could disagree with the apply order.
    pub(crate) fn enqueue(
        queue: &Arc<Self>,
        store: &Weak<CollectionStore>,
        lsn: u64,
        ops: Vec<WalRecord>,
    ) -> Result<Arc<Completion>> {
        let completion = Arc::new(Completion::new());
        let mut state = queue.state.lock();
        if state.poisoned {
            return Err(VectorSearchError::Internal(
                "commit queue failed; the collection refuses writes".to_string(),
            ));
        }
        state.queue.push_back(Ticket {
            lsn,
            ops,
            completion: Arc::clone(&completion),
        });
        state.waiters.push(Arc::clone(&completion));
        state.in_flight += 1;
        #[cfg(test)]
        {
            state.enqueued_total += 1;
        }
        if state.worker.is_none() {
            let worker = std::thread::Builder::new()
                .name("simvec-commit".to_string())
                .spawn({
                    let queue = Arc::clone(queue);
                    let store = Weak::clone(store);
                    move || worker_main(queue, store)
                })?;
            state.worker = Some(worker);
        }
        drop(state);
        queue.work.notify_all();
        Ok(completion)
    }

    /// Take every queued ticket as one batch. Returns `None` on shutdown.
    ///
    /// The test hold parks the worker outside the queue's own lock, so a test
    /// can fill the queue and then let the whole backlog flush as a single
    /// batch. It is checked before the queue is looked at, so nothing is
    /// flushed while a test holds it.
    fn take_batch(&self) -> Option<Vec<Ticket>> {
        loop {
            #[cfg(test)]
            self.hold_for_test();
            let mut state = self.state.lock();
            if state.shutdown || state.poisoned {
                return None;
            }
            if !state.queue.is_empty() {
                return Some(std::mem::take(&mut state.queue).into_iter().collect());
            }
            self.work.wait(&mut state);
        }
    }

    /// Flush `batch` with one WAL fsync, then apply it in lsn order.
    fn run_batch(&self, store: &CollectionStore, batch: Vec<Ticket>) {
        store.metrics().record_apply_queue_depth(batch.len() as u64);
        let logged: Vec<(u64, &[WalRecord])> = batch
            .iter()
            .map(|ticket| (ticket.lsn, ticket.ops.as_slice()))
            .collect();
        if let Err(e) = store.wal().append_batch(&logged) {
            // Nothing in a batch that is not durable may be applied: it would
            // exist in memory while the log could not restore it.
            let message = format!("wal group commit failed: {e}");
            for ticket in &batch {
                self.settle(&ticket.completion, Err(internal(&message)));
            }
            return;
        }
        store.metrics().record_wal_flush(batch.len());
        for ticket in &batch {
            let outcome = store.apply_queued_txn(ticket.lsn, &ticket.ops);
            let failure = outcome.as_ref().err().map(|e| e.to_string());
            self.settle(&ticket.completion, outcome);
            if let Some(message) = failure {
                // An apply error can leave the live slot numbering diverging
                // from what replay rebuilds, so the queue stops accepting
                // writes rather than risking silent corruption. Reads keep
                // working on the applied state; poisoning also settles the
                // rest of this batch, whose waiters must not hang.
                self.poison(&message);
                return;
            }
        }
    }

    /// Settle one ticket: drop its waiter registration, drop the in-flight
    /// count and wake a quiescing maintenance pass once the queue is empty.
    fn settle(&self, completion: &Arc<Completion>, outcome: Result<()>) {
        let drained = {
            let mut state = self.state.lock();
            state.waiters.retain(|w| !Arc::ptr_eq(w, completion));
            state.in_flight = state.in_flight.saturating_sub(1);
            state.in_flight == 0
        };
        if drained {
            self.drained.notify_all();
        }
        completion.settle(outcome);
    }

    /// Fail every unsettled transaction and refuse new ones.
    fn poison(&self, reason: &str) {
        let (waiters, drained) = {
            let mut state = self.state.lock();
            if state.poisoned {
                return;
            }
            state.poisoned = true;
            state.queue.clear();
            let drained = state.in_flight == 0;
            state.in_flight = 0;
            (std::mem::take(&mut state.waiters), drained)
        };
        if drained {
            self.drained.notify_all();
        }
        let message = format!("commit queue failed: {reason}");
        for waiter in waiters {
            waiter.settle(Err(internal(&message)));
        }
        self.work.notify_all();
    }

    /// Pass the queue gate: wait while a maintenance critical section holds
    /// the queue, then register as an in-coming writer.
    ///
    /// Must be taken before the store write lock, never inside it.
    pub(crate) fn begin_write(&self) -> WritePermit<'_> {
        let mut state = self.state.lock();
        while state.quiescing > 0 && !state.poisoned {
            self.work.wait(&mut state);
        }
        state.writers += 1;
        drop(state);
        WritePermit(self)
    }

    /// Close the gate and wait until the queue is empty and no writer is on
    /// its way in.
    ///
    /// A checkpoint or compaction truncates the WAL, so no transaction may be
    /// queued — or halfway through the worker — while it runs.
    pub(crate) fn quiesce(&self) -> QuiesceGuard<'_> {
        // Register first and wait second: a caller unwinding from the wait
        // still releases its registration through the guard's `Drop`, so the
        // gate cannot stay closed forever.
        let guard = QuiesceGuard::register(self);
        guard.wait_drained();
        guard
    }

    /// Stop the worker and join it. Called from `CollectionStore::drop`.
    pub(crate) fn shutdown(&self) {
        // Joining the worker from the worker thread itself would deadlock;
        // that thread is already on its way out.
        let joinable = !super::is_commit_worker();
        let worker = {
            let mut state = self.state.lock();
            state.shutdown = true;
            state.queue.clear();
            self.work.notify_all();
            state.worker.take()
        };
        if let (Some(worker), true) = (worker, joinable) {
            let _ = worker.join();
        }
    }

    /// Number of transactions queued or being applied right now.
    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        self.state.lock().in_flight
    }

    /// Whether a checkpoint / compaction critical section is holding the
    /// queue. Test-only observation hook.
    #[cfg(test)]
    pub(crate) fn quiescing(&self) -> bool {
        self.state.lock().quiescing > 0
    }

    /// Whether the worker is parked on the test hold. Test-only observation
    /// hook.
    #[cfg(test)]
    pub(crate) fn worker_parked(&self) -> bool {
        self.state.lock().parked
    }

    /// Transactions ever enqueued. Test-only observation hook.
    #[cfg(test)]
    pub(crate) fn enqueued_total(&self) -> u64 {
        self.state.lock().enqueued_total
    }
}

/// Fresh copy of a shared failure message: one error object cannot be moved
/// into several waiters.
fn internal(message: &str) -> VectorSearchError {
    VectorSearchError::Internal(message.to_string())
}

/// Worker loop: flush each backlog with one fsync, then apply it in order.
fn worker_main(queue: Arc<CommitQueue>, store: Weak<CollectionStore>) {
    // A panicking worker must not leave its waiters blocked: unwinding drops
    // this guard, which poisons the queue and fails every unsettled ticket.
    struct WorkerGuard {
        queue: Arc<CommitQueue>,
        stopped_cleanly: bool,
    }
    impl Drop for WorkerGuard {
        fn drop(&mut self) {
            if !self.stopped_cleanly {
                self.queue.poison("commit queue worker panicked");
            }
        }
    }
    let mut guard = WorkerGuard {
        queue: Arc::clone(&queue),
        stopped_cleanly: false,
    };
    set_commit_worker(true);
    loop {
        let Some(store) = store.upgrade() else {
            break;
        };
        let Some(batch) = queue.take_batch() else {
            break;
        };
        #[cfg(test)]
        if queue.request_panic_for_test() {
            panic!("test-injected commit queue panic");
        }
        set_applying(true);
        queue.run_batch(&store, batch);
        set_applying(false);
    }
    guard.stopped_cleanly = true;
}

impl CommitQueue {
    /// Test-only: ask the worker to panic before flushing its next batch.
    #[cfg(test)]
    pub(crate) fn panic_before_next_flush(&self) {
        self.state.lock().panic_before_flush = true;
    }

    /// Test-only: park the worker before it takes its next batch.
    ///
    /// Lets a test fill the queue and then let the whole backlog flush as one
    /// batch, which is the only way to assert group commit deterministically.
    /// The park is interruptible, so a dropped collection still shuts its
    /// worker down.
    #[cfg(test)]
    pub(crate) fn hold_flush(&self) {
        let mut state = self.state.lock();
        state.hold = true;
        drop(state);
        // Wake an idle worker so it reaches the park instead of waiting for
        // the next transaction.
        self.work.notify_all();
    }

    /// Test-only: release a parked worker.
    #[cfg(test)]
    pub(crate) fn release_flush(&self) {
        let mut state = self.state.lock();
        state.hold = false;
        drop(state);
        self.work.notify_all();
    }

    /// Park while the test hold is set. Runs outside the queue's own critical
    /// sections so a test can still observe the queue while the worker waits.
    #[cfg(test)]
    fn hold_for_test(&self) {
        let mut state = self.state.lock();
        state.parked = true;
        while state.hold && !state.shutdown {
            self.work.wait(&mut state);
        }
        state.parked = false;
    }

    #[cfg(test)]
    fn request_panic_for_test(&self) -> bool {
        let mut state = self.state.lock();
        let requested = state.panic_before_flush;
        state.panic_before_flush = false;
        requested
    }
}

#[cfg(test)]
#[path = "queue_tests.rs"]
mod tests;
