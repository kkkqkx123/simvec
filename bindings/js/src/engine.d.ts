export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export interface CollectionConfig extends JsonObject {
  vector_size?: number;
  distance?: string;
  index?: string;
}

export interface SearchQuery extends JsonObject {
  vector: number[] | Float32Array;
  limit: number;
}

export interface ScrollPage {
  points: JsonObject[];
  next_offset?: string | null;
}

export declare class Engine {
  constructor(dirPath: string);
  readonly closed: boolean;
  readonly lastCode: number;
  readonly lastMessage: string;
  close(): void;
  [Symbol.dispose](): void;
  createCollection(name: string, config?: CollectionConfig | null): void;
  deleteCollection(name: string): void;
  collectionExists(name: string): boolean;
  listCollections(): string[];
  collectionInfo(name: string): JsonObject;
  collectionConfig(name: string): JsonObject;
  upsert(collection: string, id: string | number, vector: number[] | Float32Array, payload?: JsonObject | null): void;
  upsertBatch(collection: string, points: JsonObject[]): void;
  delete(collection: string, id: string | number): void;
  deleteBatch(collection: string, ids: Array<string | number>): void;
  deleteByFilter(collection: string, filter: JsonObject): number;
  setPayload(collection: string, id: string | number, payload: JsonObject): void;
  mergePayload(collection: string, id: string | number, fields: JsonObject): void;
  setPayloadField(collection: string, id: string | number, key: string, value: JsonValue): void;
  applyTxn(ops: JsonObject[]): void;
  deletePayloadKeys(collection: string, id: string | number, keys: Iterable<string>): void;
  createPayloadIndex(collection: string, field: string, schema: string): void;
  deletePayloadIndex(collection: string, field: string): boolean;
  listPayloadIndexes(collection: string): Array<{ field: string; schema: string }>;
  search(collection: string, query: SearchQuery): JsonObject[];
  searchVector(
    collection: string,
    vector: number[] | Float32Array,
    limit: number,
    filter?: JsonObject | null,
    scoreThreshold?: number,
    withPayload?: boolean,
    withVector?: boolean,
  ): JsonObject[];
  get(collection: string, id: string | number): JsonObject;
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
  collectionMetrics(collection: string): JsonObject;
  runMaintenanceSweep(): void;
  promoteTempCollection(live: string, temp: string, backupSuffix: string): void;
  restorePromoteBackup(live: string): string;
  listTempCollections(): string[];
  listOldBackups(live: string): string[];
  setDefaultHnswConfig(config: JsonObject): void;
  getDefaultHnswConfig(): JsonObject | null;
  setDefaultIvfConfig(config: JsonObject): void;
  getDefaultIvfConfig(): JsonObject | null;
  setDefaultQuantizationConfig(config: JsonObject): void;
  getDefaultQuantizationConfig(): JsonObject | null;
}
