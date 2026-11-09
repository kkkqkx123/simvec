export function requireOpfsDirectory(): Promise<unknown>;
export function saveSnapshotToOpfs(engine: { exportSnapshot(): unknown }, fileName?: string): Promise<void>;
export function loadSnapshotFromOpfs(engine: { importSnapshot(snapshot: unknown): void }, fileName?: string): Promise<void>;
export function snapshotExists(fileName?: string): Promise<boolean>;
export function removeSnapshot(fileName?: string): Promise<boolean>;
export function estimateOpfsQuota(): Promise<unknown>;
export function requestOpfsPersistence(): Promise<boolean>;
export interface OpfsAutoSaveOptions {
  fileName?: string;
  debounceMs?: number;
  onError?: (error: unknown) => void;
  subscribePageHide?: boolean;
}
export interface OpfsAutoSave {
  saveNow(): Promise<void>;
  isDirty(): boolean;
  dispose(): void;
}
export function createOpfsAutoSave(
  engine: { exportSnapshot(): unknown; revision?: () => number; listCollections?: () => string[] },
  options?: OpfsAutoSaveOptions,
): OpfsAutoSave;
export function saveCollectionsToOpfs(
  engine: { exportCollection(name: string): unknown; listCollections(): string[] },
  dirName?: string,
  options?: { collections?: Array<string> },
): Promise<string[]>;
export function loadCollectionsFromOpfs(
  engine: { importCollection(snapshot: unknown): void },
  dirName?: string,
  options?: { collections?: Array<string> },
): Promise<string[]>;
