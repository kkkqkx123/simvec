export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export interface ScrollPage {
  points: JsonObject[];
  next_offset?: string | null;
}

export declare class Engine {
  static create(): Promise<Engine>;
  constructor();
  readonly closed: boolean;
  close(): void;
  [Symbol.dispose](): void;
  createCollection(name: string, config?: unknown): void;
  deleteCollection(name: string): void;
  collectionExists(name: string): boolean;
  listCollections(): string[];
  collectionInfo(name: string): unknown;
  collectionConfig(name: string): unknown;
  upsert(collection: string, id: string | number, vector: number[] | Float32Array, payload?: unknown): void;
  upsertBatch(collection: string, points: unknown[]): void;
  delete(collection: string, id: string | number): void;
  deleteBatch(collection: string, ids: Array<string | number>): void;
  deleteByFilter(collection: string, filter: unknown): number;
  applyTxn(ops: unknown[]): void;
  setPayload(collection: string, id: string | number, payload: unknown): void;
  mergePayload(collection: string, id: string | number, fields: unknown): void;
  setPayloadField(collection: string, id: string | number, key: string, value: JsonValue): void;
  deletePayloadKeys(collection: string, id: string | number, keys: string[]): void;
  createPayloadIndex(collection: string, field: string, schema: string): void;
  deletePayloadIndex(collection: string, field: string): boolean;
  listPayloadIndexes(collection: string): unknown;
  search(collection: string, query: unknown): unknown[];
  searchVector(
    collection: string,
    vector: number[] | Float32Array,
    limit: number,
    filter?: unknown,
    scoreThreshold?: number,
    withPayload?: boolean,
    withVector?: boolean,
  ): unknown[];
  get(collection: string, id: string | number): unknown;
  count(collection: string): number;
  scroll(collection: string, limit: number, offset?: string | null, withPayload?: boolean, withVector?: boolean): ScrollPage;
  buildIndex(collection: string): boolean;
  buildQuantization(collection: string): boolean;
  hasIndex(collection: string): boolean;
  hasQuantization(collection: string): boolean;
  dropIndex(collection: string): void;
  dropPublishedIndex(collection: string): void;
  compact(collection: string): number;
  drainPending(collection: string): void;
  collectionMetrics(collection: string): unknown;
  runMaintenanceSweep(): void;
  exportSnapshot(): unknown;
  importSnapshot(snapshot: unknown): void;
  revision(): number;
  exportCollection(name: string): unknown;
  importCollection(snapshot: unknown): void;
}
