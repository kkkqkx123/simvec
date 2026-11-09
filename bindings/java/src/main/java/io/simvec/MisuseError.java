package io.simvec;

/** Caller misuse: bad pointer, bad UTF-8, malformed JSON, or use of a closed engine. */
public class MisuseError extends SimvecError {
    public MisuseError(int code, String message) {
        super(code, message);
    }
}
