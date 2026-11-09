package io.simvec;

/** A collection already exists. */
public class AlreadyExistsError extends SimvecError {
    public AlreadyExistsError(int code, String message) {
        super(code, message);
    }
}
