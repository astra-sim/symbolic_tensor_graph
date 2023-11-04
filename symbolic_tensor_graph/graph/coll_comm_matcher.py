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
            if parallel_symbol in from_parallel_dims.keys():
                from_comm = from_parallel_dims[parallel_symbol]
            else:
                assert False
            if parallel_symbol in to_parallel_dims.keys():
                to_comm = to_parallel_dims[parallel_symbol]
            else:
                to_comm = (cls.EndType.REDUCED, None)
            matched_comm_pair.append((from_comm, to_comm))
        comms = list()
        for from_comm, to_comm in matched_comm_pair:
            if from_comm[0] == cls.EndType.PARTITION:
                if to_comm[0] == cls.EndType.PARTITION:
                    if not from_comm[1] == to_comm[1]:
                        # TODO: need some clever way to examine if there two dims contains each other and no a2a required
                        comms.append(
                            (cls.CommType.ALL_TO_ALL, from_comm[1], to_comm[1])
                        )
                    else:
                        # do nothing
                        pass
                elif to_comm[0] == cls.EndType.REDUCED:
                    assert to_comm[1] is None  # shouldnt dim on reduced dim
                    comms.append((cls.CommType.ALL_GATHER, from_comm[1], to_comm[1]))
                else:
                    assert False
            elif from_comm[0] == cls.EndType.REDUCED:
                if to_comm[0] == cls.EndType.PARTITION:
                    comms.append(
                        (cls.CommType.REDUCE_SCATTER, from_comm[1], to_comm[1])
                    )
                elif to_comm[0] == cls.EndType.REDUCED:
                    if to_comm[1] is None:
                        comms.append(
                            (cls.CommType.ALL_REDUCE, from_comm[1], to_comm[1])
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
            matched = None
            for parallel_symbol in remaining_parallel_symbols:
                if parallel_symbol in dim.free_symbols:
                    matched = parallel_symbol
                    break
            if not matched is None:
                remaining_parallel_symbols.remove(parallel_symbol)
                parallel_dims[matched] = cls.EndType.PARTITION, dim

        for dim in hidden:
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
