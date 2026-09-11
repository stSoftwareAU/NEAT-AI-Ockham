//! The run's time source (Issue #214).
//!
//! Every duration the optimisation loop reasons about is read from one
//! injected [`Clock`]: the deadline it stops on, and the scorer costs its
//! budget arithmetic is sized from — the cohort sizing of Issue #58 and the
//! screening reserve of Issue #77. Production runs on [`SystemClock`], the
//! real monotonic clock.
//!
//! The boundary is the budget: local work the run only *reports* on — the
//! exact-cleanup pre-pass timing — stays on the real clock, because no budget
//! decision reads it and a manual clock would report real work as free.
//!
//! A test drives [`ManualClock`] instead, so a budget decision is asserted
//! against a time source the test controls rather than raced against the wall
//! clock: the scripted scorer spends the budget by advancing the clock, and
//! the same decision is reached on a loaded CI runner, a shared laptop and ARM
//! alike.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A monotonic time source for the run's budget arithmetic.
pub trait Clock: Send + Sync {
    /// The current instant.
    fn now(&self) -> Instant;

    /// How long has passed since `earlier`, saturating at zero.
    fn since(&self, earlier: Instant) -> Duration {
        self.now().saturating_duration_since(earlier)
    }

    /// Milliseconds since `earlier`, saturating at zero.
    fn ms_since(&self, earlier: Instant) -> u64 {
        self.since(earlier).as_millis() as u64
    }
}

/// The real monotonic clock — what every run outside a test uses.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// A clock that moves only when it is told to (tests).
///
/// Cheap to clone: every handle shares one reading, so a scripted scorer can
/// spend the run's budget on the same clock the loop reads its deadline from.
#[derive(Debug, Clone)]
pub struct ManualClock {
    state: Arc<ManualState>,
}

#[derive(Debug)]
struct ManualState {
    base: Instant,
    /// Nanoseconds this clock has been advanced past [`ManualState::base`].
    advanced_ns: AtomicU64,
}

impl ManualClock {
    /// A clock reading zero elapsed.
    pub fn new() -> Self {
        Self {
            state: Arc::new(ManualState {
                base: Instant::now(),
                advanced_ns: AtomicU64::new(0),
            }),
        }
    }

    /// Move the clock forward, saturating rather than wrapping.
    ///
    /// Saturation matters: a wrapped reading would run a deadline backwards
    /// and turn an expired budget into a fresh one, which is exactly the
    /// silent fault this clock exists to remove.
    pub fn advance(&self, by: Duration) {
        let ns = u64::try_from(by.as_nanos()).unwrap_or(u64::MAX);
        let _ =
            self.state
                .advanced_ns
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                    Some(current.saturating_add(ns))
                });
    }

    /// How far this clock has been advanced.
    pub fn elapsed(&self) -> Duration {
        Duration::from_nanos(self.state.advanced_ns.load(Ordering::SeqCst))
    }
}

impl Default for ManualClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Instant {
        self.state.base + self.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_system_clock_moves_forward() {
        let clock = SystemClock;
        let first = clock.now();
        let second = clock.now();
        assert!(second >= first, "a monotonic clock never runs backwards");
        assert_eq!(
            clock.since(second + Duration::from_secs(60)),
            Duration::ZERO,
            "a reading ahead of now saturates at zero rather than panicking"
        );
    }

    #[test]
    fn a_manual_clock_stands_still_until_it_is_advanced() {
        let clock = ManualClock::new();
        let opened = clock.now();
        assert_eq!(clock.elapsed(), Duration::ZERO);
        assert_eq!(clock.now(), opened, "nothing advanced it, so nothing moved");

        clock.advance(Duration::from_millis(100));
        assert_eq!(clock.since(opened), Duration::from_millis(100));
        assert_eq!(clock.ms_since(opened), 100);

        clock.advance(Duration::from_millis(400));
        assert_eq!(clock.elapsed(), Duration::from_millis(500));
    }

    #[test]
    fn every_handle_shares_one_reading() {
        let clock = ManualClock::new();
        let spender = clock.clone();
        let opened = clock.now();
        spender.advance(Duration::from_secs(2));
        assert_eq!(
            clock.ms_since(opened),
            2_000,
            "a clone spends the same budget the original reads"
        );
    }

    #[test]
    fn an_absurd_advance_saturates_rather_than_wrapping() {
        let clock = ManualClock::new();
        clock.advance(Duration::from_secs(u64::MAX));
        clock.advance(Duration::from_secs(1));
        assert_eq!(
            clock.elapsed(),
            Duration::from_nanos(u64::MAX),
            "a saturated clock stays at its ceiling; it must never wrap to zero"
        );
    }
}
