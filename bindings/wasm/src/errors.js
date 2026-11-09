export const SIMVEC_OK = 0;
export const SIMVEC_ERROR = 1;
export const SIMVEC_INTERNAL = 2;
export const SIMVEC_NOMEM = 7;
export const SIMVEC_IOERR = 10;
export const SIMVEC_CORRUPT = 11;
export const SIMVEC_NOTFOUND = 12;
export const SIMVEC_ALREADY_EXISTS = 13;
export const SIMVEC_INVALID_CONFIG = 16;
export const SIMVEC_INVALID_DIM = 19;
export const SIMVEC_MISUSE = 20;
export const SIMVEC_NON_FINITE = 21;
export const SIMVEC_CAPACITY_EXCEEDED = 22;
export const SIMVEC_FILTER = 23;

export class SimvecError extends Error {
  constructor(message, code) {
    super(message);
    this.name = "SimvecError";
    this.code = code;
  }
}

export class NotFoundError extends SimvecError {
  constructor(message, code) {
    super(message, code);
    this.name = "NotFoundError";
  }
}

export class AlreadyExistsError extends SimvecError {
  constructor(message, code) {
    super(message, code);
    this.name = "AlreadyExistsError";
  }
}

export class InvalidConfigError extends SimvecError {
  constructor(message, code) {
    super(message, code);
    this.name = "InvalidConfigError";
  }
}

export class InvalidDimensionError extends SimvecError {
  constructor(message, code) {
    super(message, code);
    this.name = "InvalidDimensionError";
  }
}

export class MisuseError extends SimvecError {
  constructor(message, code) {
    super(message, code);
    this.name = "MisuseError";
  }
}

export class CapacityExceededError extends SimvecError {
  constructor(message, code) {
    super(message, code);
    this.name = "CapacityExceededError";
  }
}

export class FilterError extends SimvecError {
  constructor(message, code) {
    super(message, code);
    this.name = "FilterError";
  }
}

const CODE_TO_ERROR = new Map([
  [SIMVEC_NOTFOUND, NotFoundError],
  [SIMVEC_ALREADY_EXISTS, AlreadyExistsError],
  [SIMVEC_INVALID_CONFIG, InvalidConfigError],
  [SIMVEC_INVALID_DIM, InvalidDimensionError],
  [SIMVEC_MISUSE, MisuseError],
  [SIMVEC_CAPACITY_EXCEEDED, CapacityExceededError],
  [SIMVEC_FILTER, FilterError],
]);

export function raiseForCode(code, message) {
  if (code === SIMVEC_OK) {
    return;
  }
  const Kind = CODE_TO_ERROR.get(code) ?? SimvecError;
  throw new Kind(`[simvec:${code}] ${message || "unknown error"}`, code);
}

function messageFromThrown(thrown) {
  if (typeof thrown === "string") {
    return thrown;
  }
  if (thrown && typeof thrown.message === "string" && thrown.message.length > 0) {
    return thrown.message;
  }
  return String(thrown ?? "");
}

export function raiseForWasm(thrown) {
  const message = messageFromThrown(thrown);
  const match = message.match(/\[simvec:(\d+)\]/);
  const code = match ? Number(match[1]) : SIMVEC_ERROR;
  const Kind = CODE_TO_ERROR.get(code) ?? SimvecError;
  throw new Kind(message, code);
}
