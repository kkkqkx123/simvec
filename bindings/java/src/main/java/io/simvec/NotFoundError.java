package io.simvec;

/** A collection or point was not found. */
public class NotFoundError extends SimvecError {
    public NotFoundError(int code, String message) {
        super(code, message);
    }
}
