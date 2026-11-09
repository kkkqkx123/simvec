package simvec

import (
	"errors"
	"fmt"
)

// Status codes shared with the C API.
const (
	CodeOK               = 0
	CodeError            = 1
	CodeInternal         = 2
	CodeNoMem            = 7
	CodeIO               = 10
	CodeCorrupt          = 11
	CodeNotFound         = 12
	CodeAlreadyExists    = 13
	CodeInvalidConfig    = 16
	CodeInvalidDim       = 19
	CodeMisuse           = 20
	CodeNonFinite        = 21
	CodeCapacityExceeded = 22
	CodeFilter           = 23
)

// SimvecError is the base error carrying the raw status code.
type SimvecError struct {
	Code    int
	Message string
}

func (e *SimvecError) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("[simvec:%d] unknown error", e.Code)
	}
	return fmt.Sprintf("[simvec:%d] %s", e.Code, e.Message)
}

// StatusCode reports the raw C API status code.
func (e *SimvecError) StatusCode() int { return e.Code }

// NotFoundError reports a missing collection or point.
type NotFoundError struct{ SimvecError }

// AlreadyExistsError reports a duplicate collection.
type AlreadyExistsError struct{ SimvecError }

// InvalidConfigError reports a rejected collection or engine config.
type InvalidConfigError struct{ SimvecError }

// InvalidDimensionError reports a vector size mismatch.
type InvalidDimensionError struct{ SimvecError }

// MisuseError reports a null pointer, invalid UTF-8, malformed caller JSON,
// or use of a closed engine.
type MisuseError struct{ SimvecError }

// CapacityExceededError reports capacity exhaustion.
type CapacityExceededError struct{ SimvecError }

// FilterError reports a rejected filter expression.
type FilterError struct{ SimvecError }

func (e *NotFoundError) Unwrap() error         { return &e.SimvecError }
func (e *AlreadyExistsError) Unwrap() error    { return &e.SimvecError }
func (e *InvalidConfigError) Unwrap() error    { return &e.SimvecError }
func (e *InvalidDimensionError) Unwrap() error { return &e.SimvecError }
func (e *MisuseError) Unwrap() error           { return &e.SimvecError }
func (e *CapacityExceededError) Unwrap() error { return &e.SimvecError }
func (e *FilterError) Unwrap() error           { return &e.SimvecError }

func newError(code int, message string) error {
	if message == "" {
		message = "unknown error"
	}
	switch code {
	case CodeNotFound:
		return &NotFoundError{SimvecError{Code: code, Message: message}}
	case CodeAlreadyExists:
		return &AlreadyExistsError{SimvecError{Code: code, Message: message}}
	case CodeInvalidConfig:
		return &InvalidConfigError{SimvecError{Code: code, Message: message}}
	case CodeInvalidDim:
		return &InvalidDimensionError{SimvecError{Code: code, Message: message}}
	case CodeMisuse:
		return &MisuseError{SimvecError{Code: code, Message: message}}
	case CodeCapacityExceeded:
		return &CapacityExceededError{SimvecError{Code: code, Message: message}}
	case CodeFilter:
		return &FilterError{SimvecError{Code: code, Message: message}}
	default:
		return &SimvecError{Code: code, Message: message}
	}
}

// CodeOf extracts the raw status code from a binding error,
// returning -1 when the error did not originate here.
func CodeOf(err error) int {
	var base *SimvecError
	if errors.As(err, &base) {
		return base.Code
	}
	return -1
}

// IsNotFound reports whether err is a not-found failure.
func IsNotFound(err error) bool { return CodeOf(err) == CodeNotFound }

// IsAlreadyExists reports whether err is a duplicate-collection failure.
func IsAlreadyExists(err error) bool { return CodeOf(err) == CodeAlreadyExists }

// IsInvalidConfig reports whether err is a bad-config failure.
func IsInvalidConfig(err error) bool { return CodeOf(err) == CodeInvalidConfig }

// IsInvalidDimension reports whether err is a dimension-mismatch failure.
func IsInvalidDimension(err error) bool { return CodeOf(err) == CodeInvalidDim }

// IsMisuse reports whether err is a caller-misuse failure.
func IsMisuse(err error) bool { return CodeOf(err) == CodeMisuse }

// IsCapacityExceeded reports whether err is a capacity-exhaustion failure.
func IsCapacityExceeded(err error) bool { return CodeOf(err) == CodeCapacityExceeded }

// IsFilter reports whether err is a filter failure.
func IsFilter(err error) bool { return CodeOf(err) == CodeFilter }
