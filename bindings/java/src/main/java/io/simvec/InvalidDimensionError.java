package io.simvec;

/** A vector size did not match the collection. */
public class InvalidDimensionError extends SimvecError {
    public InvalidDimensionError(int code, String message) {
        super(code, message);
    }
}
