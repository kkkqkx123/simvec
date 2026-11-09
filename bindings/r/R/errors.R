simvec_code_of <- function(code) {
  as.integer(code)
}

stop_simvec <- function(code, message) {
  code <- as.integer(code)
  if (is.null(message) || is.na(message) || !nzchar(message)) {
    message <- "unknown error"
  }
  text <- sprintf("[simvec:%d] %s", code, message)
  cls <- switch(as.character(code),
    "12" = "simvec_not_found",
    "13" = "simvec_already_exists",
    "16" = "simvec_invalid_config",
    "19" = "simvec_invalid_dimension",
    "20" = "simvec_misuse",
    "22" = "simvec_capacity_exceeded",
    "23" = "simvec_filter",
    "simvec_error"
  )
  cond <- structure(
    list(message = text, code = code),
    class = c(cls, "simvec_error", "error", "condition")
  )
  stop(cond)
}

raise_if_failed <- function(res) {
  code <- as.integer(res$code)
  if (code != 0L) {
    stop_simvec(code, res$errmsg)
  }
  invisible(res$payload)
}

payload_or_raise <- function(res) {
  code <- as.integer(res$code)
  if (code != 0L) {
    stop_simvec(code, res$errmsg)
  }
  res$payload
}

#' Extract the raw status code from a simvec condition.
#'
#' Returns -1 when the condition did not originate here.
#' @export
simvec_code <- function(err) {
  if (inherits(err, "simvec_error") && !is.null(err$code)) {
    return(as.integer(err$code))
  }
  -1L
}

#' @export
is_simvec_error <- function(err) {
  inherits(err, "simvec_error")
}

is_code <- function(err, code) {
  is_simvec_error(err) && identical(simvec_code(err), as.integer(code))
}

#' @export
is_not_found <- function(err) {
  is_code(err, 12L)
}

#' @export
is_already_exists <- function(err) {
  is_code(err, 13L)
}

#' @export
is_invalid_config <- function(err) {
  is_code(err, 16L)
}

#' @export
is_invalid_dimension <- function(err) {
  is_code(err, 19L)
}

#' @export
is_misuse <- function(err) {
  is_code(err, 20L)
}

#' @export
is_capacity_exceeded <- function(err) {
  is_code(err, 22L)
}

#' @export
is_filter <- function(err) {
  is_code(err, 23L)
}
