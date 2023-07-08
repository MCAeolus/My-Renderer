use std::time::Instant;

pub struct AvgTimer {
    timer: Instant,
    num_entries: u32,
    total: f64,
}

pub trait TimerTrait {
    fn start(&mut self) -> ();
    fn elapse(&mut self) -> ();
    fn record<F: Fn() -> ()>(&mut self, f: F) -> ();
    fn average(&self) -> (u32, f64);
    fn total(&self) -> f64;
    fn report(&self) -> String;
}

impl TimerTrait for AvgTimer {
    #[inline]
    fn start(&mut self) {
        self.timer = Instant::now();
    }
    #[inline]
    fn elapse(&mut self) {
        self.num_entries += 1;
        self.total += (Instant::now() - self.timer).as_secs_f64();
    }
    fn average(&self) -> (u32, f64) {
        (self.num_entries, self.total / self.num_entries as f64)
    }
    #[inline]
    fn record<F: Fn() -> ()>(&mut self, f: F) -> () {
        self.start();
        f();
        self.elapse();
    }
    fn total(&self) -> f64 {
        self.total
    }
    fn report(&self) -> String {
        let avg = self.average();
        format!("Total time: {}s, Average per run: {}s, Total entries: {}", self.total(), avg.1, avg.0)
    }
}

pub fn new() -> AvgTimer {
    AvgTimer { timer: Instant::now(), num_entries: 0, total: 0., }
}
