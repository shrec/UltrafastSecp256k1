## Description

<!-- Brief summary of changes -->

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Performance improvement
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update
- [ ] CI/Build infrastructure

## Changes Made

<!-- List key changes -->

-

## Testing

- [ ] All existing tests pass (`ctest --test-dir build --output-on-failure`)
- [ ] New tests added (if applicable)
- [ ] Benchmark results attached (if performance-related)

## Hot Path Checklist (if applicable)

- [ ] Zero heap allocations in hot path
- [ ] No strings/iostreams/exceptions in hot path
- [ ] Explicit in/out/scratch buffers
- [ ] No `%` or `/` where Montgomery/Barrett is available

## Additional Notes

<!-- Any other context about the PR -->
