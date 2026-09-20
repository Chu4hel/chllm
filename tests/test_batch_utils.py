from chllm.utils import split_batch


def test_split_batch_even():
    batch = [1, 2, 3, 4]
    part1, part2 = split_batch(batch)
    assert part1 == [1, 2]
    assert part2 == [3, 4]


def test_split_batch_odd():
    batch = [1, 2, 3]
    part1, part2 = split_batch(batch)
    assert part1 == [1]
    assert part2 == [2, 3]


def test_split_batch_single():
    batch = [1]
    part1, part2 = split_batch(batch)
    assert part1 == [1]
    assert part2 == []
