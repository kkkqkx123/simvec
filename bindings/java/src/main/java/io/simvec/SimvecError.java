package io.simvec;

/**
 * Base error carrying the raw C API status code.
 *
 * <p>Seven status codes raise typed subclasses; all other non-zero codes
 * raise this base with the numeric code kept in the message.
 */
public class SimvecError extends RuntimeException {
    public static final int OK = 0;
    public static final int ERROR = 1;
    public static final int INTERNAL = 2;
    public static final int NOMEM = 7;
    public static final int IOERR = 10;
    public static final int CORRUPT = 11;
    public static final int NOTFOUND = 12;
    public static final int ALREADY_EXISTS = 13;
    public static final int INVALID_CONFIG = 16;
    public static final int INVALID_DIM = 19;
    public static final int MISUSE = 20;
    public static final int NON_FINITE = 21;
    public static final int CAPACITY_EXCEEDED = 22;
    public static final int FILTER = 23;

    private final int code;

    public SimvecError(int code, String message) {
        super("[simvec:" + code + "] " + (message == null || message.isEmpty() ? "unknown error" : message));
        this.code = code;
    }

    /** Raw C API status code. */
    public int getCode() {
        return code;
    }

    /** Raise the typed exception matching a non-zero status code. */
    public static SimvecError raiseForCode(int code, String message) {
        switch (code) {
            case NOTFOUND:
                return new NotFoundError(code, message);
            case ALREADY_EXISTS:
                return new AlreadyExistsError(code, message);
            case INVALID_CONFIG:
                return new InvalidConfigError(code, message);
            case INVALID_DIM:
                return new InvalidDimensionError(code, message);
            case MISUSE:
                return new MisuseError(code, message);
            case CAPACITY_EXCEEDED:
                return new CapacityExceededError(code, message);
            case FILTER:
                return new FilterError(code, message);
            default:
                return new SimvecError(code, message);
        }
    }

    /** Extract the raw status code from a binding error, or -1 when foreign. */
    public static int codeOf(Throwable err) {
        if (err instanceof SimvecError) {
            return ((SimvecError) err).getCode();
        }
        return -1;
    }

    public static boolean isNotFound(Throwable err) {
        return codeOf(err) == NOTFOUND;
    }

    public static boolean isAlreadyExists(Throwable err) {
        return codeOf(err) == ALREADY_EXISTS;
    }

    public static boolean isInvalidConfig(Throwable err) {
        return codeOf(err) == INVALID_CONFIG;
    }

    public static boolean isInvalidDimension(Throwable err) {
        return codeOf(err) == INVALID_DIM;
    }

    public static boolean isMisuse(Throwable err) {
        return codeOf(err) == MISUSE;
    }

    public static boolean isCapacityExceeded(Throwable err) {
        return codeOf(err) == CAPACITY_EXCEEDED;
    }

    public static boolean isFilter(Throwable err) {
        return codeOf(err) == FILTER;
    }
}
