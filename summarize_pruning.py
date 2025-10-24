import re

print("=" * 70)
print("Isolation Pruning Summary - Data2 Training")
print("=" * 70)

with open("data2_training_log.txt") as f:
    log = f.read()

# Find all isolation pruning messages
pruning_msgs = re.findall(r'\[Isolation Pruning\] Removing (\d+) isolated points', log)
pruning_counts = [int(x) for x in pruning_msgs]

# Find any skipped messages
skipped_msgs = re.findall(r'\[Isolation Pruning\] Skipped.*?(\d+)/(\d+)', log)

print(f"\nIsolation Pruning Statistics:")
print(f"  Total pruning events: {len(pruning_counts)}")
print(f"  Total points removed: {sum(pruning_counts)}")
print(f"  Average per event: {sum(pruning_counts)/len(pruning_counts) if pruning_counts else 0:.1f}")
print(f"  Max removed in single event: {max(pruning_counts) if pruning_counts else 0}")
print(f"  Min removed in single event: {min(pruning_counts) if pruning_counts else 0}")

print(f"\nSafety Check (>20% threshold):")
print(f"  Times triggered: {len(skipped_msgs)}")
if skipped_msgs:
    print(f"  ⚠️  Safety check was triggered:")
    for would_remove, total in skipped_msgs[:5]:  # Show first 5
        print(f"    - Would remove {would_remove}/{total} points ({int(would_remove)/int(total)*100:.1f}%)")
else:
    print(f"  ✅ Safety check never triggered - pruning always under control!")

# Check histogram of removal counts
print(f"\nPruning Event Distribution:")
ranges = [(0, 20), (20, 50), (50, 100), (100, 200), (200, float('inf'))]
for low, high in ranges:
    count = sum(1 for x in pruning_counts if low <= x < high)
    if count > 0:
        pct = count / len(pruning_counts) * 100
        print(f"  {low}-{high if high != float('inf') else '+'}  points: {count} events ({pct:.1f}%)")

