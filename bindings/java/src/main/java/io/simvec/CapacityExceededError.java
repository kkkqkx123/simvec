package io.simvec;

/** Capacity was exhausted. */
public class CapacityExceededError extends SimvecError {
    public CapacityExceededError(int code, String message) {
        super(code, message);
    }
}
