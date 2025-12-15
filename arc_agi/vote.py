from arc_agi.types import ARCAGIResult, ExpertConfig
from arc_agi.utils import _mean_soft, canonical_test_key


def rank_results(results: list[ARCAGIResult], config: ExpertConfig) -> list[ARCAGIResult]:
    """
    Groups results by identical test outputs, creates buckets for passers/failures,
    and ranks them according to the configuration (voting, diversity, tie-breaks).
    """
    if not results:
        return []

    use_new_voting = config.get("use_new_voting", True)
    count_failed_matches = config.get("count_failed_matches", True)
    iters_tiebreak = config.get("iters_tiebreak", False)
    low_to_high_iters = config.get("low_to_high_iters", False)

    # Buckets
    candidate_buckets: dict[str, list[ARCAGIResult]] = {}
    failure_buckets: dict[str, list[ARCAGIResult]] = {}

    for res in results:
        # Determine if it is a passer
        is_passer = False
        train_results = res.get("train_results", [])
        if train_results:
            is_passer = all(rr.get("success", False) for rr in train_results)

        key = canonical_test_key(res.get("results", []))

        if is_passer:
            candidate_buckets.setdefault(key, []).append(res)
        else:
            failure_buckets.setdefault(key, []).append(res)

    if use_new_voting:
        # Optionally merge failures into passers if outputs match
        if count_failed_matches:
            for k in list(failure_buckets.keys()):
                if k in candidate_buckets:
                    candidate_buckets[k].extend(failure_buckets[k])
                    del failure_buckets[k]

        # ---- Passers: sort by vote count desc; diversity-first ----
        passer_groups: list[list[ARCAGIResult]] = list(candidate_buckets.values())

        if iters_tiebreak:
            # Sort within groups
            passer_groups = [
                sorted(ps, key=lambda x: x.get('iteration', 999), reverse=not low_to_high_iters)
                for ps in passer_groups
            ]
            # Sort groups by best iteration
            passer_groups = sorted(passer_groups, key=lambda x: x[0].get('iteration', 999), reverse=low_to_high_iters)

        # Sort passers by how many votes they have.
        passer_groups = sorted(passer_groups, key=len, reverse=True)

        ordered: list[ARCAGIResult] = []
        # one per group for diversity
        ordered.extend([grp[0] for grp in passer_groups if grp])

        # ---- Failures: grouped + ranked ----
        # within each failure group, best first by mean soft_score desc
        for fs in failure_buckets.values():
            fs.sort(key=_mean_soft, reverse=True)

        failure_groups: list[list[ARCAGIResult]] = list(failure_buckets.values())
        # Sort groups: votes (desc), tie-break by best member's mean soft_score (desc)
        failure_groups.sort(
            key=lambda fs: (len(fs), _mean_soft(fs[0]) if fs else 0.0),
            reverse=True,
        )

        # diversity-first over failure groups
        ordered.extend([fs[0] for fs in failure_groups if fs])
        # remaining passer members
        ordered.extend([m for grp in passer_groups for m in grp[1:]])
        # remaining failure members
        ordered.extend([m for fs in failure_groups for m in fs[1:]])

        return ordered

    else:
        # ---- Old mode ----
        passer_groups: list[list[ARCAGIResult]] = sorted(
            candidate_buckets.values(), key=len, reverse=True
        )

        firsts = [grp[0] for grp in passer_groups if grp]

        failed_flat: list[ARCAGIResult] = [
            r for fs in failure_buckets.values() for r in fs
        ]
        failed_sorted = sorted(failed_flat, key=_mean_soft, reverse=True)

        rest = [m for grp in passer_groups for m in grp[1:]]

        return firsts + failed_sorted + rest
