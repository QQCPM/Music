# Codebase Cleanup Plan

## Files to Archive (Outdated/Superseded)

### OUTDATED - Phase 0 Intermediate Work (Before Discovery)
These files were written when we thought emotions were NOT encoded:

1. **NEXT_STEPS_FINAL.md** - Outdated (assumed weak signal, but we found 96% accuracy!)
2. **SOLUTION_STRATEGY.md** - Outdated (searching for signal we already found)
3. **CRITICAL_REALITY_CHECK.md** - Outdated (0.47% signal, but that was transformer - T5 has 6.6%!)
4. **DEBUG_SUMMARY.md** - Outdated (activation bug fix, already documented elsewhere)
5. **ACTIVATION_EXTRACTION_FIX.md** - Outdated (bug fix documented in summary)
6. **CODEBASE_CLEANUP_SUMMARY.md** - Outdated (previous cleanup)
7. **PHASE0_COMPLETE_PLAN.md** - Outdated (old Phase 0 plan, superseded by PHASE0_TO_PHASE1_SUMMARY)
8. **DEEP_ANALYSIS_FUNDAMENTAL_ISSUES.md** - Outdated (led to T5 discovery, now documented in summary)

### OLD TEST SCRIPTS (One-Time Use)
9. **debug_activation_extraction.py** - One-time debugging script (bug fixed)
10. **test_fixed_extractor.py** - Validation script (bug verified fixed)
11. **validate_results_critically.py** - Critical analysis (led to T5 discovery, done)

### KEEP - Current Valid Documentation
- **INDEX.md** - Navigation hub
- **README.md** - Project overview
- **SETUP_COMPLETE.md** - Setup summary
- **PHASE0_TO_PHASE1_SUMMARY.md** - Complete Phase 0 results
- **PHASE1_QUICKSTART.md** - Quick start guide
- **PHASE1_ROADMAP.md** - Detailed 3-week plan
- **SYSTEM_OVERVIEW.md** - Technical architecture
- **START_HERE.md** - Codebase navigation
- **test_text_embeddings.py** - Key discovery test (T5 embeddings)

## Action Plan

1. Move outdated files to archive/phase0_development/
2. Keep only current, accurate documentation
3. Verify all remaining docs are consistent
4. Create simple START_HERE for returning user
