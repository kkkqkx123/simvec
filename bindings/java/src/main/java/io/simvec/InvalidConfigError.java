package io.simvec;

/** A collection or engine config was rejected. */
public class InvalidConfigError extends SimvecError {
    public InvalidConfigError(int code, String message) {
        super(code, message);
    }
}
