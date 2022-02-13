import numpy as np

from dsec import DSEC

from argparse import ArgumentParser


def main() :
    parser = ArgumentParser()

    parser.add_argument("seqs", nargs='+', type=int)
    parser.add_argument("--bin_type", type=str, default='interpolation')
    parser.add_argument("--num_bins", type=int, default=15)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    print(args.seqs)

    dsec = DSEC(seqs=args.seqs, add_backward=True,
                num_bins=args.num_bins, bin_type=args.bin_type)

    dsec.write_binning_np(range(len(dsec.seq_lens)), args.name)

    dsec_written = DSEC(event_set=args.name,
                        seqs=args.seqs, add_backward=True,
                        num_bins=args.num_bins,
                        bin_type=args.bin_type)

    #e_live = dsec.prep_item(0, 186, True)['events']
    #e_written = dsec_written.prep_item(0, 185, False)['events']
    #e_written = dsec_written.prep_item(0, 186, True)['events']


    """    for i in range(len(dsec) // 2 - 1) :
        print("forward backward test frame " + str(i))
        eflive_data = dsec.prep_item(0, i, False)
        eflive = eflive_data['events']
        efliveback_data = dsec.prep_item(0, i+1, True)
        efliveback = efliveback_data['events']

        assert efliveback_data['dt'] == eflive_data['dt']

        fail = (np.abs(eflive[:, :2] - np.flip(efliveback[:, :2], axis=0)) > 1e-6).any(axis=1)
        eflivefail =  eflive[fail, :]
        eflivebackfail = np.flip(efliveback, axis=0)[fail, :]

        assert (eflive[:, :2] == np.flip(efliveback[:, :2], axis=0)).all()

        eflive_sort_idx = np.argsort(np.around(eflive[:, 2], 6), # np.array([round(eflive[i, 2], 6) for i in range(len(eflive))]),
                                     kind='mergesort')
        eflive_sorted = eflive[eflive_sort_idx, :]

        # TODO do the flipping
        eflivebackflip = np.flip(efliveback[:, :], axis=0)
        assert(eflive[:, :2] == eflivebackflip[:, :2]).all()
        t_reconstruct = eflivebackflip[:, 2]
        t_reconstruct -= efliveback_data['dt']
        t_reconstruct *= -1
        error_bool = (np.around(eflive[:, 2], 5) * 10000).astype(int) != (np.around(t_reconstruct, 5) * 10000).astype(int)
        error_bool = np.array([round(eflive[i, 2], 4) for i in range(len(eflive))]) != np.array([round(t_reconstruct[i], 4) for i in range(len(t_reconstruct))])
        #for i in range(10) :
        #    print("{:.50f}".format(t_reconstruct[error_bool][i]))
        #    print("{:.50f}".format(eflive[error_bool, 2][i]))
        efliveback_sort_idx = np.argsort(np.around(t_reconstruct, 6), #np.array([round(t_reconstruct[i], 12) for i in range(len(t_reconstruct))]),
                                         kind='mergesort')
        assert(efliveback_sort_idx == eflive_sort_idx).all()
        efliveback_sorted = eflivebackflip[efliveback_sort_idx, :]
        assert(efliveback_sorted[:, :2] == eflive_sorted[:, :2]).all()
        efliveback_sorted = np.flip(efliveback_sorted, axis=0)


        efwrite = dsec_written.prep_item(0, i, False)['events']
        efwriteback = dsec_written.prep_item(0, i+1, True)['events']

        fail = (np.abs(efwrite[:, :2] - np.flip(efwriteback[:, :2], axis=0)) > 1e-6).any(axis=1)
        efwritefail = efwrite[fail, :]
        efwritebackfail = np.flip(efwrite, axis=0)[fail, :]

        assert (efwrite[:, :2] == np.flip(efwriteback[:, :2], axis=0)).all()

        writelive_error = np.abs(efwrite - eflive_sorted)
        assert (writelive_error < 1e-6).all()
        assert (efwrite[:, :2] == eflive_sorted[:, :2]).all()


        writeliveback_error = np.abs(efwriteback - efliveback_sorted)
        assert(writeliveback_error < 1e-6).all()
        assert(efwriteback[:, :2] == efliveback_sorted[:, :2]).all()"""

    # (0, 186, True)
    for i in range(len(dsec)) :
        print("frame " + str(i))

        e_live_data = dsec[i]
        e_live = e_live_data['events']
        if args.bin_type == 'sum' :
            if dsec.get_local_idx(i)[2] :
                e_live = np.flip(e_live, axis=0)
                t_reconstruct = -e_live[:, 2] + e_live_data['dt']
                sort_idx = np.argsort(np.around(t_reconstruct, 9),
                                      kind='mergesort')
                e_live = e_live[sort_idx, :]
                e_live = np.flip(e_live, axis=0)
            else :
                sort_idx = np.argsort(np.around(e_live[:, 2], 9),
                                      kind='mergesort')
                e_live = e_live[sort_idx, :]
            # e_live = np.flip(e_live, axis=0)
        e_written = dsec_written[i]['events']

        if (len(e_live) != len(e_written)) :
            print("differing number of events")
            print(dsec.get_local_idx(i))
            print(len(e_live))
            print(len(e_written))
            continue

        diff = np.abs(e_live - e_written)
        if not (diff < 1e-6).all() :
            error_lines = (diff >= 1e-6).any(axis=1)
            e_live_error = e_live[error_lines, :]
            e_written_error = e_written[error_lines, :]
            print(e_live[error_lines, :])
            print(e_written[error_lines, :])
            print(error_lines.sum())

if __name__ == "__main__" :
    main()