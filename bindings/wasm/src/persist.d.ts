export function requireOpfsDirectory(): Promise<unknown>;
export function saveSnapshotToOpfs(engine: { exportSnapshot(): unknown }, fileName?: string): Promise<void>;
export function loadSnapshotFromOpfs(engine: { importSnapshot(snapshot: unknown): void }, fileName?: string): Promise<void>;
