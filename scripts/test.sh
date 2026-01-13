#!/bin/bash
# Run all tests across the monorepo

set -e

echo "🧪 Running all tests..."
echo ""

PACKAGES=(
    "packages/runtime"
    "packages/langchain"
    "packages/langgraph"
    "packages/openai"
    "packages/anthropic"
    "packages/llamaindex"
)

TOTAL_PASSED=0
TOTAL_FAILED=0

for pkg in "${PACKAGES[@]}"; do
    if [ -d "$pkg/tests" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📦 Testing: $pkg"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        cd "$pkg"
        if uv run --extra dev pytest -v --tb=short; then
            echo "✅ $pkg passed"
        else
            echo "❌ $pkg failed"
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
        fi
        cd - > /dev/null
        echo ""
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $TOTAL_FAILED -eq 0 ]; then
    echo "🎉 All tests passed!"
else
    echo "❌ $TOTAL_FAILED package(s) had failures"
    exit 1
fi
