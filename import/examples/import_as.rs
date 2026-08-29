// Import a snapshot directory under an explicit output name — exactly the
// call cyb's models world makes.
fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: import_as <snapshot-dir> <name>");
    let name = args.next().expect("usage: import_as <snapshot-dir> <name>");
    match import::pipeline::import_snapshot(&dir, &name) {
        Ok(p) => println!("OK {}", p.display()),
        Err(e) => {
            eprintln!("FAIL {e}");
            std::process::exit(1);
        }
    }
}
