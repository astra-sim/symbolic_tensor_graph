import copy


class CommunicationMatcher:

    class EndType:
        PARTITION = "partition"
        REDUCED = "reduced"

    class CommType:
        ALL_GATHER = "all_gather"
        ALL_REDUCE = "all_reduce"
        ALL_TO_ALL = "all_to_all"
        REDUCE_SCATTER = "reduce_scatter"

    @classmethod
    def match_comms(
        cls, from_shape, from_hidden, to_shape, to_hidden, parallel_symbols
    ):
        from_parallel_dims = cls.get_parallel_dims(
            from_shape, from_hidden, parallel_symbols
        )
        to_parallel_dims = cls.get_parallel_dims(to_shape, to_hidden, parallel_symbols)
        matched_comm_pair = list()
        for parallel_symbol in parallel_symbols:
            if (not parallel_symbol in from_parallel_dims.keys()) and (
                not parallel_symbol in to_parallel_dims.keys()
            ):
                continue
            if parallel_symbol in from_parallel_dims.keys():
                from_comm = from_parallel_dims[parallel_symbol]
            else:
                assert False
            if parallel_symbol in to_parallel_dims.keys():
                to_comm = to_parallel_dims[parallel_symbol]
            else:
                to_comm = (cls.EndType.REDUCED, None)
            matched_comm_pair.append((from_comm, to_comm, parallel_symbol))
        comms = list()
        for from_comm, to_comm, parallel_symbol in matched_comm_pair:
            if from_comm[0] == cls.EndType.PARTITION:
                if to_comm[0] == cls.EndType.PARTITION:
                    if not from_comm[1] == to_comm[1]:
                        # TODO: need some clever way to examine if there two dims contains each other and no a2a required
                        comms.append(
                            (
                                cls.CommType.ALL_TO_ALL,
                                from_comm[1],
                                to_comm[1],
                                parallel_symbol,
                            )
                        )
                    else:
                        # do nothing
                        pass
                elif to_comm[0] == cls.EndType.REDUCED:
                    assert to_comm[1] is None  # shouldnt dim on reduced dim
                    comms.append(
                        (
                            cls.CommType.ALL_GATHER,
                            from_comm[1],
                            to_comm[1],
                            parallel_symbol,
                        )
                    )
                else:
                    assert False
            elif from_comm[0] == cls.EndType.REDUCED:
                if to_comm[0] == cls.EndType.PARTITION:
                    comms.append(
                        (
                            cls.CommType.REDUCE_SCATTER,
                            from_comm[1],
                            to_comm[1],
                            parallel_symbol,
                        )
                    )
                elif to_comm[0] == cls.EndType.REDUCED:
                    if to_comm[1] is None:
                        comms.append(
                            (
                                cls.CommType.ALL_REDUCE,
                                from_comm[1],
                                to_comm[1],
                                parallel_symbol,
                            )
                        )
                    else:
                        # do nothing
                        pass
                else:
                    assert False
            else:
                assert False
        return comms

    @classmethod
    def get_parallel_dims(cls, shape, hidden, parallel_symbols):
        remaining_parallel_symbols = copy.deepcopy(parallel_symbols)

        parallel_dims = dict()

        for dim in shape:
            if isinstance(dim, int) or isinstance(dim, float):
                continue
            matched = None
            for parallel_symbol in remaining_parallel_symbols:
                if parallel_symbol in dim.free_symbols:
                    matched = parallel_symbol
                    break
            if not matched is None:
                remaining_parallel_symbols.remove(matched)
                parallel_dims[matched] = cls.EndType.PARTITION, dim

        for dim in hidden:
            if isinstance(dim, int) or isinstance(dim, float):
                continue
            matched = None
            for parallel_symbol in remaining_parallel_symbols:
                if parallel_symbol in dim.free_symbols:
                    matched = parallel_symbol
                    break
            if not matched is None:
                remaining_parallel_symbols.remove(parallel_symbol)
                parallel_dims[matched] = cls.EndType.REDUCED, dim

        assert len(parallel_dims) + len(remaining_parallel_symbols) == len(
            parallel_symbols
        )
        return parallel_dims


class CommunicationMatcherV2:
    class EndType:
        DUPLICATED = "duplicated"
        PARTITIONED = "partitioned"
        PARTIALSUM = "partialsum"

    class CommType:
        ALL_GATHER = "all_gather"
        ALL_REDUCE = "all_reduce"
        ALL_TO_ALL = "all_to_all"
        REDUCE_SCATTER = "reduce_scatter"
        SLICED = "sliced"  # virtual comm type, not really a coll communication
        GATHER_SCATTER = (
            "gather_scatter"  # virtual comm type, general case of a2a/Identity
        )
        IDENTITY = "identity"  # virtual comm type, doing nothing

    @classmethod
    def get_parallel_dims(cls, shape, hidden, parallel_symbols):
        remaining_parallel_symbols = copy.deepcopy(parallel_symbols)

        parallel_dims = dict()
        for dim in shape:
            if isinstance(dim, int) or isinstance(dim, float):
                continue
            matched = None
            for parallel_symbol in remaining_parallel_symbols:
                if parallel_symbol in dim.free_symbols:
                    matched = parallel_symbol
                    break
            if not matched is None:
                remaining_parallel_symbols.remove(matched)
                parallel_dims[matched] = cls.EndType.PARTITIONED, dim

        for dim in hidden:
            if isinstance(dim, int) or isinstance(dim, float):
                continue
            matched = None
            for parallel_symbol in remaining_parallel_symbols:
                if parallel_symbol in dim.free_symbols:
                    matched = parallel_symbol
                    break
            if not matched is None:
                remaining_parallel_symbols.remove(parallel_symbol)
                parallel_dims[matched] = cls.EndType.PARTIALSUM, dim

        for symbol in remaining_parallel_symbols:
            parallel_dims[symbol] = cls.EndType.DUPLICATED, None

        return parallel_dims

    @classmethod
    def match_comms(
        cls, from_shape, from_hidden, to_shape, to_hidden, parallel_symbols
    ):
        from_parallel_dims = cls.get_parallel_dims(
            from_shape, from_hidden, parallel_symbols
        )
        to_parallel_dims = cls.get_parallel_dims(to_shape, to_hidden, parallel_symbols)
        matched_comm_pair = list()
        for parallel_symbol in parallel_symbols:
            from_parallel_dim = from_parallel_dims[parallel_symbol]
            to_parallel_dim = to_parallel_dims[parallel_symbol]
            matched_comm_pair.append(
                (from_parallel_dim, to_parallel_dim, parallel_symbol)
            )

        comms = list()
        for from_comm, to_comm, parallel_symbol in matched_comm_pair:
            if to_comm[0] == cls.EndType.PARTIALSUM:
                if from_comm[0] == cls.EndType.PARTIALSUM:
                    # no change
                    comms.append(
                        (
                            cls.CommType.IDENTITY,
                            from_comm[1],
                            to_comm[1],
                            parallel_symbol,
                        )
                    )
                else:
                    assert False, "cannot produce partialsum from non-partialsum"
            elif to_comm[0] == cls.EndType.PARTITIONED:
                if from_comm[0] == cls.EndType.DUPLICATED:
                    # from duplicated to partition, slices
                    comms.append(
                        (cls.CommType.SLICED, from_comm[1], to_comm[1], parallel_symbol)
                    )
                elif from_comm[0] == cls.EndType.PARTITIONED:
                    comms.append(
                        (
                            cls.CommType.GATHER_SCATTER,
                            from_comm[1],
                            to_comm[1],
                            parallel_symbol,
                        )
                    )
                elif from_comm[0] == cls.EndType.PARTIALSUM:
                    # partialsum to partition
                    comms.append(
                        (
                            cls.CommType.REDUCE_SCATTER,
                            from_comm[1],
                            to_comm[1],
                            parallel_symbol,
                        )
                    )
                else:
                    assert False
            elif to_comm[0] == cls.EndType.DUPLICATED:
                if from_comm[0] == cls.EndType.DUPLICATED:
                    comms.append(
                        (
                            cls.CommType.IDENTITY,
                            from_comm[1],
                            to_comm[1],
                            parallel_symbol,
                        )
                    )
                elif from_comm[0] == cls.EndType.PARTITIONED:
                    comms.append(
                        (
                            cls.CommType.ALL_GATHER,
                            from_comm[1],
                            to_comm[1],
                            parallel_symbol,
                        )
                    )
                elif from_comm[0] == cls.EndType.PARTIALSUM:
                    comms.append(
                        (
                            cls.CommType.ALL_REDUCE,
                            from_comm[1],
                            to_comm[1],
                            parallel_symbol,
                        )
                    )
                else:
                    assert False
            else:
                assert False

        # special case handling
        for i, comm in enumerate(comms):
            if comm[0] == cls.CommType.GATHER_SCATTER:
                from_dim = comm[1]
                to_dim = comm[2]
                if from_dim == to_dim:
                    comm = (cls.CommType.IDENTITY,) + comm[1:]
                else:
                    comm = (cls.CommType.ALL_TO_ALL,) + comm[1:]
                comms[i] = comm

        def _filter_fn(comm):
            if comm[0] in {cls.CommType.SLICED}:
                print(
                    f"unefficient collective {comm[0]} found! check if the sharding plan is reasonable."
                )
            return comm[0] not in {cls.CommType.IDENTITY, cls.CommType.SLICED}

        filtered_comms = filter(
            _filter_fn,
            comms,
        )
        return filtered_comms
