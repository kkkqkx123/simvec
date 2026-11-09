export declare const SIMVEC_OK: 0;
export declare const SIMVEC_ERROR: 1;
export declare const SIMVEC_INTERNAL: 2;
export declare const SIMVEC_NOMEM: 7;
export declare const SIMVEC_IOERR: 10;
export declare const SIMVEC_CORRUPT: 11;
export declare const SIMVEC_NOTFOUND: 12;
export declare const SIMVEC_ALREADY_EXISTS: 13;
export declare const SIMVEC_INVALID_CONFIG: 16;
export declare const SIMVEC_INVALID_DIM: 19;
export declare const SIMVEC_MISUSE: 20;
export declare const SIMVEC_NON_FINITE: 21;
export declare const SIMVEC_CAPACITY_EXCEEDED: 22;
export declare const SIMVEC_FILTER: 23;

export class SimvecError extends Error {
  code: number;
  constructor(message: string, code: number);
}
export class NotFoundError extends SimvecError {}
export class AlreadyExistsError extends SimvecError {}
export class InvalidConfigError extends SimvecError {}
export class InvalidDimensionError extends SimvecError {}
export class MisuseError extends SimvecError {}
export class CapacityExceededError extends SimvecError {}
export class FilterError extends SimvecError {}

export function raiseForCode(code: number, message: string): void;
