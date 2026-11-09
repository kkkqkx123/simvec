package io.simvec;

/** A filter expression was rejected. */
public class FilterError extends SimvecError {
    public FilterError(int code, String message) {
        super(code, message);
    }
}
