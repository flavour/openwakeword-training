#!/bin/bash
# Train all three wakeword candidates for Freyja.
# Run inside Docker: docker compose run --rm trainer bash train_all.sh
#
# Each model takes 4-8 hours, so this is an overnight job.
# Models output to my_custom_model/<name>.onnx

set -e

DATA_DIR="${DATA_DIR:-/app/data}"

echo "============================================"
echo "  Freyja Wakeword Training - All Candidates"
echo "============================================"
echo ""
echo "Training 3 models: freyja, hey_freyja, okay_freyja"
echo "Estimated time: 12-24 hours total"
echo ""

# Train each candidate
for WORD in "freyja" "hey freyja" "okay freyja"; do
    SAFE=$(echo "$WORD" | tr ' ' '_')
    echo ""
    echo "========================================"
    echo "  Training: $WORD"
    echo "  Started: $(date)"
    echo "========================================"
    echo ""

    python train.py \
        --wake-word "$WORD" \
        --samples-per-voice 200 \
        --training-steps 50000 \
        --layer-size 64 \
        --data-dir "$DATA_DIR"

    echo ""
    echo "  Finished $WORD at $(date)"

    # Check if model was produced
    if [ -f "my_custom_model/${SAFE}.onnx" ]; then
        SIZE=$(du -h "my_custom_model/${SAFE}.onnx" | cut -f1)
        echo "  ✓ Model: my_custom_model/${SAFE}.onnx ($SIZE)"
    else
        echo "  ✗ WARNING: Model not found!"
    fi
done

echo ""
echo "============================================"
echo "  All training complete! $(date)"
echo "============================================"
echo ""
echo "Models:"
ls -lh my_custom_model/*.onnx 2>/dev/null || echo "  No models found :("
echo ""
echo "Test with:"
echo "  python test_model.py --model my_custom_model/freyja.onnx"
echo "  python test_model.py --model my_custom_model/hey_freyja.onnx"
echo "  python test_model.py --model my_custom_model/okay_freyja.onnx"
