package simvec

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../target/debug -L${SRCDIR}/../../target/release -lsimvec -Wl,-rpath,${SRCDIR}/../../target/debug -Wl,-rpath,${SRCDIR}/../../target/release
#include <stdlib.h>
#include "simvec.h"
*/
import "C"

import (
	"encoding/json"
	"runtime"
	"sync"
	"unsafe"
)

// PayloadIndex describes one payload index on a collection.
type PayloadIndex struct {
	Field  string `json:"field"`
	Schema string `json:"schema"`
}

// ScrollPage is one paginated scan result. A nil NextOffset means the scan
// started without a cursor is unaffected; exhaustion is signaled by a short
// page carrying fewer points than requested.
type ScrollPage struct {
	Points     []map[string]any `json:"points"`
	NextOffset *string          `json:"next_offset"`
}

// Engine owns one native engine handle rooted at a directory.
type Engine struct {
	mu     sync.RWMutex
	handle *C.struct_simvec_engine_t
	closed bool
}

// NewEngine opens the engine rooted at dir.
func NewEngine(dir string) (*Engine, error) {
	cDir := C.CString(dir)
	defer C.free(unsafe.Pointer(cDir))
	var out *C.struct_simvec_engine_t
	if code := C.simvec_open(cDir, &out); code != C.int(CodeOK) {
		return nil, openError(int(code))
	}
	engine := &Engine{handle: out}
	runtime.SetFinalizer(engine, func(target *Engine) { _ = target.Close() })
	return engine, nil
}

// Close releases the native handle. Safe to call more than once.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	handle := e.handle
	e.handle = nil
	if handle != nil {
		if code := C.simvec_close(handle); code != C.int(CodeOK) {
			return newError(int(code), "close failed")
		}
	}
	return nil
}

// Closed reports whether the engine has been closed.
func (e *Engine) Closed() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.closed
}

// LastCode reports the most recent status code on the native handle.
func (e *Engine) LastCode() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return CodeMisuse
	}
	return int(C.simvec_errcode(e.handle))
}

// LastMessage reports the most recent error text without raising.
func (e *Engine) LastMessage() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return "engine is closed"
	}
	var out *C.char
	if C.simvec_last_error(e.handle, &out) != C.int(CodeOK) {
		return ""
	}
	return takeString(out)
}

// Float32s converts a float64 slice into the float32 form vectors use.
func Float32s(vector []float64) []float32 {
	out := make([]float32, len(vector))
	for i, value := range vector {
		out[i] = float32(value)
	}
	return out
}

func openError(code int) error {
	text := C.simvec_last_error_text()
	return newError(code, takeString(text))
}

func (e *Engine) fail(code C.int) error {
	if code == C.int(CodeOK) {
		return nil
	}
	var out *C.char
	if C.simvec_last_error(e.handle, &out) == C.int(CodeOK) {
		return newError(int(code), takeString(out))
	}
	return newError(int(code), "")
}

func takeString(out *C.char) string {
	if out == nil {
		return ""
	}
	text := C.GoString(out)
	C.simvec_free_string(out)
	return text
}

func boolToInt(value bool) C.int {
	if value {
		return 1
	}
	return 0
}

func jsonBytes(value any) ([]byte, error) {
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, newError(CodeMisuse, "value is not JSON serializable")
	}
	return raw, nil
}

func cString(text string) *C.char {
	return C.CString(text)
}

func cJSONOrNull(raw []byte, present bool) *C.char {
	if !present {
		return nil
	}
	return C.CString(string(raw))
}

func vectorParts(vector []float32) (*C.float, C.size_t) {
	if len(vector) == 0 {
		return nil, 0
	}
	return (*C.float)(unsafe.Pointer(&vector[0])), C.size_t(len(vector))
}

func decodeText(out *C.char, target any) error {
	text := takeString(out)
	if err := json.Unmarshal([]byte(text), target); err != nil {
		return newError(CodeInternal, "failed to decode engine response")
	}
	return nil
}

// CreateCollection creates a collection. A nil config leaves engine defaults in place.
func (e *Engine) CreateCollection(name string, config any) error {
	var raw []byte
	var present bool
	if config != nil {
		body, err := jsonBytes(config)
		if err != nil {
			return err
		}
		raw, present = body, true
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cName := cString(name)
	defer C.free(unsafe.Pointer(cName))
	cConfig := cJSONOrNull(raw, present)
	if cConfig != nil {
		defer C.free(unsafe.Pointer(cConfig))
	}
	return e.fail(C.simvec_create_collection(e.handle, cName, cConfig))
}

// DeleteCollection deletes a collection by name.
func (e *Engine) DeleteCollection(name string) error {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cName := cString(name)
	defer C.free(unsafe.Pointer(cName))
	return e.fail(C.simvec_delete_collection(e.handle, cName))
}

// CollectionExists reports whether a collection exists.
func (e *Engine) CollectionExists(name string) (bool, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return false, newError(CodeMisuse, "engine is closed")
	}
	cName := cString(name)
	defer C.free(unsafe.Pointer(cName))
	var out C.int
	if err := e.fail(C.simvec_collection_exists(e.handle, cName, &out)); err != nil {
		return false, err
	}
	return out != 0, nil
}

// ListCollections lists collection names.
func (e *Engine) ListCollections() ([]string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	var out *C.char
	if err := e.fail(C.simvec_list_collections(e.handle, &out)); err != nil {
		return nil, err
	}
	var names []string
	if err := decodeText(out, &names); err != nil {
		return nil, err
	}
	if names == nil {
		names = []string{}
	}
	return names, nil
}

// CollectionInfo returns the info document of a collection.
func (e *Engine) CollectionInfo(name string) (map[string]any, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	cName := cString(name)
	defer C.free(unsafe.Pointer(cName))
	var out *C.char
	if err := e.fail(C.simvec_collection_info(e.handle, cName, &out)); err != nil {
		return nil, err
	}
	var info map[string]any
	if err := decodeText(out, &info); err != nil {
		return nil, err
	}
	return info, nil
}

// CollectionConfig returns the config document of a collection.
func (e *Engine) CollectionConfig(name string) (map[string]any, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	cName := cString(name)
	defer C.free(unsafe.Pointer(cName))
	var out *C.char
	if err := e.fail(C.simvec_collection_config(e.handle, cName, &out)); err != nil {
		return nil, err
	}
	var config map[string]any
	if err := decodeText(out, &config); err != nil {
		return nil, err
	}
	return config, nil
}

// PromoteTempCollection promotes a temp rebuild collection over a live name.
func (e *Engine) PromoteTempCollection(live, temp, backupSuffix string) error {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cLive := cString(live)
	defer C.free(unsafe.Pointer(cLive))
	cTemp := cString(temp)
	defer C.free(unsafe.Pointer(cTemp))
	cSuffix := cString(backupSuffix)
	defer C.free(unsafe.Pointer(cSuffix))
	return e.fail(C.simvec_promote_temp_collection(e.handle, cLive, cTemp, cSuffix))
}

// RestorePromoteBackup restores the newest promote backup, returning its name.
func (e *Engine) RestorePromoteBackup(live string) (string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return "", newError(CodeMisuse, "engine is closed")
	}
	cLive := cString(live)
	defer C.free(unsafe.Pointer(cLive))
	var out *C.char
	if err := e.fail(C.simvec_restore_promote_backup(e.handle, cLive, &out)); err != nil {
		return "", err
	}
	var name string
	if err := decodeText(out, &name); err != nil {
		return "", err
	}
	return name, nil
}

// ListTempCollections lists temp rebuild collections.
func (e *Engine) ListTempCollections() ([]string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	var out *C.char
	if err := e.fail(C.simvec_list_temp_collections(e.handle, &out)); err != nil {
		return nil, err
	}
	var names []string
	if err := decodeText(out, &names); err != nil {
		return nil, err
	}
	if names == nil {
		names = []string{}
	}
	return names, nil
}

// ListOldBackups lists promote backups of one live collection.
func (e *Engine) ListOldBackups(live string) ([]string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	cLive := cString(live)
	defer C.free(unsafe.Pointer(cLive))
	var out *C.char
	if err := e.fail(C.simvec_list_old_backups(e.handle, cLive, &out)); err != nil {
		return nil, err
	}
	var names []string
	if err := decodeText(out, &names); err != nil {
		return nil, err
	}
	if names == nil {
		names = []string{}
	}
	return names, nil
}

// Upsert inserts or replaces one point. A nil payload stores no payload.
func (e *Engine) Upsert(collection, id string, vector []float32, payload any) error {
	var raw []byte
	var present bool
	if payload != nil {
		body, err := jsonBytes(payload)
		if err != nil {
			return err
		}
		raw, present = body, true
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cID := cString(id)
	defer C.free(unsafe.Pointer(cID))
	cPayload := cJSONOrNull(raw, present)
	if cPayload != nil {
		defer C.free(unsafe.Pointer(cPayload))
	}
	data, size := vectorParts(vector)
	return e.fail(C.simvec_upsert(e.handle, cCollection, cID, data, size, cPayload))
}

// UpsertBatch inserts or replaces points described as id/vector/payload objects.
func (e *Engine) UpsertBatch(collection string, points any) error {
	raw, err := jsonBytes(points)
	if err != nil {
		return err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cPoints := cString(string(raw))
	defer C.free(unsafe.Pointer(cPoints))
	return e.fail(C.simvec_upsert_batch(e.handle, cCollection, cPoints))
}

// Delete removes one point by id.
func (e *Engine) Delete(collection, id string) error {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cID := cString(id)
	defer C.free(unsafe.Pointer(cID))
	return e.fail(C.simvec_delete(e.handle, cCollection, cID))
}

// DeleteBatch removes points by id.
func (e *Engine) DeleteBatch(collection string, ids []string) error {
	raw, err := jsonBytes(ids)
	if err != nil {
		return err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cIDs := cString(string(raw))
	defer C.free(unsafe.Pointer(cIDs))
	return e.fail(C.simvec_delete_batch(e.handle, cCollection, cIDs))
}

// DeleteByFilter removes points matching a filter, returning the removed count.
func (e *Engine) DeleteByFilter(collection string, filter any) (uint64, error) {
	raw, err := jsonBytes(filter)
	if err != nil {
		return 0, err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return 0, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cFilter := cString(string(raw))
	defer C.free(unsafe.Pointer(cFilter))
	var out C.uint64_t
	if err := e.fail(C.simvec_delete_by_filter(e.handle, cCollection, cFilter, &out)); err != nil {
		return 0, err
	}
	return uint64(out), nil
}

// SetPayload replaces the whole payload of one point.
func (e *Engine) SetPayload(collection, id string, payload any) error {
	raw, err := jsonBytes(payload)
	if err != nil {
		return err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cID := cString(id)
	defer C.free(unsafe.Pointer(cID))
	cPayload := cString(string(raw))
	defer C.free(unsafe.Pointer(cPayload))
	return e.fail(C.simvec_set_payload(e.handle, cCollection, cID, cPayload))
}

// MergePayload merges fields into the payload of one point.
func (e *Engine) MergePayload(collection, id string, fields any) error {
	raw, err := jsonBytes(fields)
	if err != nil {
		return err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cID := cString(id)
	defer C.free(unsafe.Pointer(cID))
	cFields := cString(string(raw))
	defer C.free(unsafe.Pointer(cFields))
	return e.fail(C.simvec_merge_payload(e.handle, cCollection, cID, cFields))
}

// DeletePayloadKeys removes payload fields from one point.
func (e *Engine) DeletePayloadKeys(collection, id string, keys []string) error {
	raw, err := jsonBytes(keys)
	if err != nil {
		return err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cID := cString(id)
	defer C.free(unsafe.Pointer(cID))
	cKeys := cString(string(raw))
	defer C.free(unsafe.Pointer(cKeys))
	return e.fail(C.simvec_delete_payload_keys(e.handle, cCollection, cID, cKeys))
}

// SetPayloadField writes one payload field of one point.
func (e *Engine) SetPayloadField(collection, id, key string, value any) error {
	raw, err := jsonBytes(value)
	if err != nil {
		return err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cID := cString(id)
	defer C.free(unsafe.Pointer(cID))
	cKey := cString(key)
	defer C.free(unsafe.Pointer(cKey))
	cValue := cString(string(raw))
	defer C.free(unsafe.Pointer(cValue))
	return e.fail(C.simvec_set_payload_field(e.handle, cCollection, cID, cKey, cValue))
}

// ApplyTxn applies upsert/delete operations grouped per collection.
func (e *Engine) ApplyTxn(ops any) error {
	raw, err := jsonBytes(ops)
	if err != nil {
		return err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cOps := cString(string(raw))
	defer C.free(unsafe.Pointer(cOps))
	return e.fail(C.simvec_apply_txn(e.handle, cOps))
}

// CreatePayloadIndex creates a payload index with the given schema name.
func (e *Engine) CreatePayloadIndex(collection, field, schema string) error {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cField := cString(field)
	defer C.free(unsafe.Pointer(cField))
	cSchema := cString(schema)
	defer C.free(unsafe.Pointer(cSchema))
	return e.fail(C.simvec_create_payload_index(e.handle, cCollection, cField, cSchema))
}

// DeletePayloadIndex deletes a payload index, reporting whether one existed.
func (e *Engine) DeletePayloadIndex(collection, field string) (bool, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return false, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cField := cString(field)
	defer C.free(unsafe.Pointer(cField))
	var out C.int
	if err := e.fail(C.simvec_delete_payload_index(e.handle, cCollection, cField, &out)); err != nil {
		return false, err
	}
	return out != 0, nil
}

// ListPayloadIndexes lists payload indexes of a collection.
func (e *Engine) ListPayloadIndexes(collection string) ([]PayloadIndex, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	var out *C.char
	if err := e.fail(C.simvec_list_payload_indexes(e.handle, cCollection, &out)); err != nil {
		return nil, err
	}
	var indexes []PayloadIndex
	if err := decodeText(out, &indexes); err != nil {
		return nil, err
	}
	if indexes == nil {
		indexes = []PayloadIndex{}
	}
	return indexes, nil
}

// Search runs a full query document and returns matching hits.
func (e *Engine) Search(collection string, query any) ([]map[string]any, error) {
	raw, err := jsonBytes(query)
	if err != nil {
		return nil, err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cQuery := cString(string(raw))
	defer C.free(unsafe.Pointer(cQuery))
	var out *C.char
	if err := e.fail(C.simvec_search(e.handle, cCollection, cQuery, &out)); err != nil {
		return nil, err
	}
	var hits []map[string]any
	if err := decodeText(out, &hits); err != nil {
		return nil, err
	}
	if hits == nil {
		hits = []map[string]any{}
	}
	return hits, nil
}

// SearchVector runs a vector query without building a query document.
// A nil filter disables filtering and a negative scoreThreshold disables
// the similarity cutoff.
func (e *Engine) SearchVector(collection string, vector []float32, limit uint64, filter any, scoreThreshold float32, withPayload, withVector bool) ([]map[string]any, error) {
	var raw []byte
	var present bool
	if filter != nil {
		body, err := jsonBytes(filter)
		if err != nil {
			return nil, err
		}
		raw, present = body, true
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cFilter := cJSONOrNull(raw, present)
	if cFilter != nil {
		defer C.free(unsafe.Pointer(cFilter))
	}
	data, size := vectorParts(vector)
	var out *C.char
	code := C.simvec_search_vector(e.handle, cCollection, data, size, C.uint64_t(limit), cFilter, C.float(scoreThreshold), boolToInt(withPayload), boolToInt(withVector), &out)
	if err := e.fail(code); err != nil {
		return nil, err
	}
	var hits []map[string]any
	if err := decodeText(out, &hits); err != nil {
		return nil, err
	}
	if hits == nil {
		hits = []map[string]any{}
	}
	return hits, nil
}

// Get fetches one point by id.
func (e *Engine) Get(collection, id string) (map[string]any, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cID := cString(id)
	defer C.free(unsafe.Pointer(cID))
	var out *C.char
	if err := e.fail(C.simvec_get(e.handle, cCollection, cID, &out)); err != nil {
		return nil, err
	}
	var point map[string]any
	if err := decodeText(out, &point); err != nil {
		return nil, err
	}
	return point, nil
}

// CountByFilter returns the number of live points matching a filter. Pure read.
func (e *Engine) CountByFilter(collection string, filter any) (uint64, error) {
	raw, err := jsonBytes(filter)
	if err != nil {
		return 0, err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return 0, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	cFilter := cString(string(raw))
	defer C.free(unsafe.Pointer(cFilter))
	var out C.uint64_t
	if err := e.fail(C.simvec_count_by_filter(e.handle, cCollection, cFilter, &out)); err != nil {
		return 0, err
	}
	return uint64(out), nil
}

// Count returns the number of live points in a collection.
func (e *Engine) Count(collection string) (uint64, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return 0, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	var out C.uint64_t
	if err := e.fail(C.simvec_count(e.handle, cCollection, &out)); err != nil {
		return 0, err
	}
	return uint64(out), nil
}

// Scroll scans one page. A nil offset starts from the beginning.
func (e *Engine) Scroll(collection string, limit uint64, offset *string, withPayload, withVector bool) (*ScrollPage, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	var cOffset *C.char
	if offset != nil {
		cOffset = cString(*offset)
		defer C.free(unsafe.Pointer(cOffset))
	}
	var out *C.char
	code := C.simvec_scroll(e.handle, cCollection, C.uint64_t(limit), cOffset, boolToInt(withPayload), boolToInt(withVector), &out)
	if err := e.fail(code); err != nil {
		return nil, err
	}
	var page ScrollPage
	if err := decodeText(out, &page); err != nil {
		return nil, err
	}
	if page.Points == nil {
		page.Points = []map[string]any{}
	}
	return &page, nil
}

// ScrollByFilter scans one filtered page. Limit counts only matching points;
// page until an empty page because a short page does not imply exhaustion.
// A nil offset starts from the beginning.
func (e *Engine) ScrollByFilter(collection string, filter any, limit uint64, offset *string, withPayload, withVector bool) (*ScrollPage, error) {
	raw, err := jsonBytes(filter)
	if err != nil {
		return nil, err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	var cOffset *C.char
	if offset != nil {
		cOffset = cString(*offset)
		defer C.free(unsafe.Pointer(cOffset))
	}
	cFilter := cString(string(raw))
	defer C.free(unsafe.Pointer(cFilter))
	var out *C.char
	code := C.simvec_scroll_by_filter(e.handle, cCollection, C.uint64_t(limit), cOffset, cFilter, boolToInt(withPayload), boolToInt(withVector), &out)
	if err := e.fail(code); err != nil {
		return nil, err
	}
	var page ScrollPage
	if err := decodeText(out, &page); err != nil {
		return nil, err
	}
	if page.Points == nil {
		page.Points = []map[string]any{}
	}
	return &page, nil
}

// BuildIndex builds the vector index, reporting whether work was done.
func (e *Engine) BuildIndex(collection string) (bool, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return false, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	var out C.int
	if err := e.fail(C.simvec_build_index(e.handle, cCollection, &out)); err != nil {
		return false, err
	}
	return out != 0, nil
}

// BuildQuantization builds quantization, reporting whether work was done.
func (e *Engine) BuildQuantization(collection string) (bool, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return false, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	var out C.int
	if err := e.fail(C.simvec_build_quantization(e.handle, cCollection, &out)); err != nil {
		return false, err
	}
	return out != 0, nil
}

// HasIndex reports whether a vector index exists.
func (e *Engine) HasIndex(collection string) (bool, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return false, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	var out C.int
	if err := e.fail(C.simvec_has_index(e.handle, cCollection, &out)); err != nil {
		return false, err
	}
	return out != 0, nil
}

// HasQuantization reports whether quantization exists.
func (e *Engine) HasQuantization(collection string) (bool, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return false, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	var out C.int
	if err := e.fail(C.simvec_has_quantization(e.handle, cCollection, &out)); err != nil {
		return false, err
	}
	return out != 0, nil
}

// DropIndex drops the vector index.
func (e *Engine) DropIndex(collection string) error {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	return e.fail(C.simvec_drop_index(e.handle, cCollection))
}

// DropPublishedIndex drops the published index only.
func (e *Engine) DropPublishedIndex(collection string) error {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	return e.fail(C.simvec_drop_published_index(e.handle, cCollection))
}

// Compact reclaims deleted points, returning the live count.
func (e *Engine) Compact(collection string) (uint64, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return 0, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	var out C.uint64_t
	if err := e.fail(C.simvec_compact(e.handle, cCollection, &out)); err != nil {
		return 0, err
	}
	return uint64(out), nil
}

// DrainPending waits for queued maintenance work on a collection.
func (e *Engine) DrainPending(collection string) error {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	return e.fail(C.simvec_drain_pending(e.handle, cCollection))
}

// CollectionMetrics returns the metrics snapshot of a collection.
func (e *Engine) CollectionMetrics(collection string) (map[string]any, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	cCollection := cString(collection)
	defer C.free(unsafe.Pointer(cCollection))
	var out *C.char
	if err := e.fail(C.simvec_collection_metrics(e.handle, cCollection, &out)); err != nil {
		return nil, err
	}
	var metrics map[string]any
	if err := decodeText(out, &metrics); err != nil {
		return nil, err
	}
	return metrics, nil
}

// RunMaintenanceSweep runs a global maintenance sweep.
func (e *Engine) RunMaintenanceSweep() error {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	return e.fail(C.simvec_run_maintenance_sweep(e.handle))
}

// SetDefaultHnswConfig sets the engine-wide default HNSW config.
func (e *Engine) SetDefaultHnswConfig(config any) error {
	raw, err := jsonBytes(config)
	if err != nil {
		return err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cConfig := cString(string(raw))
	defer C.free(unsafe.Pointer(cConfig))
	return e.fail(C.simvec_set_default_hnsw_config(e.handle, cConfig))
}

// GetDefaultHnswConfig returns the engine-wide default HNSW config, or nil when unset.
func (e *Engine) GetDefaultHnswConfig() (map[string]any, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	var out *C.char
	if err := e.fail(C.simvec_get_default_hnsw_config(e.handle, &out)); err != nil {
		return nil, err
	}
	var config map[string]any
	if err := decodeText(out, &config); err != nil {
		return nil, err
	}
	return config, nil
}

// SetDefaultIvfConfig sets the engine-wide default IVF config.
func (e *Engine) SetDefaultIvfConfig(config any) error {
	raw, err := jsonBytes(config)
	if err != nil {
		return err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cConfig := cString(string(raw))
	defer C.free(unsafe.Pointer(cConfig))
	return e.fail(C.simvec_set_default_ivf_config(e.handle, cConfig))
}

// GetDefaultIvfConfig returns the engine-wide default IVF config, or nil when unset.
func (e *Engine) GetDefaultIvfConfig() (map[string]any, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	var out *C.char
	if err := e.fail(C.simvec_get_default_ivf_config(e.handle, &out)); err != nil {
		return nil, err
	}
	var config map[string]any
	if err := decodeText(out, &config); err != nil {
		return nil, err
	}
	return config, nil
}

// SetDefaultQuantizationConfig sets the engine-wide default quantization config.
func (e *Engine) SetDefaultQuantizationConfig(config any) error {
	raw, err := jsonBytes(config)
	if err != nil {
		return err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return newError(CodeMisuse, "engine is closed")
	}
	cConfig := cString(string(raw))
	defer C.free(unsafe.Pointer(cConfig))
	return e.fail(C.simvec_set_default_quantization_config(e.handle, cConfig))
}

// GetDefaultQuantizationConfig returns the engine-wide default quantization config, or nil when unset.
func (e *Engine) GetDefaultQuantizationConfig() (map[string]any, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.handle == nil {
		return nil, newError(CodeMisuse, "engine is closed")
	}
	var out *C.char
	if err := e.fail(C.simvec_get_default_quantization_config(e.handle, &out)); err != nil {
		return nil, err
	}
	var config map[string]any
	if err := decodeText(out, &config); err != nil {
		return nil, err
	}
	return config, nil
}
