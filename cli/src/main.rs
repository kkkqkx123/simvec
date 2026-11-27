fn main() {
    std::process::exit(simvec_cli::run(std::env::args().skip(1).collect()));
}
