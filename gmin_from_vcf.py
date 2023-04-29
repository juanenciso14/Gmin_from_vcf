#!/usr/bin/env python

###################################
# Author: Juan Enciso - Nov. 2022 #
# Burg Lab. ULeth.                #
###################################

import egglib as egg
import itertools as itt
import numpy as np
import argparse
import sys


def set_structure(pop_file, vcfhandle):
    """
    :param pop file: path to a text file relating samples and populations
    :param vcfhandle: instance of egglib.io.VcfParser
    Return the appropriate structure dictionary.
    """
    samples = [vcfhandle.get_sample(i) for i in range(vcfhandle.num_samples)]

    dict_pop_sample = {}

    with open(pop_file, "r") as pophandle:
        lines = pophandle.readlines()
        for line in lines:
            ind, pop = line.strip().split()
            ind_tidx = samples.index(ind) * 2
            if dict_pop_sample.get(pop) == None:
                dict_pop_sample[pop] = {ind: [ind_tidx, ind_tidx + 1]}
            else:
                dict_pop_sample[pop][ind] = [ind_tidx, ind_tidx + 1]

    return egg.struct_from_dict({"pops": dict_pop_sample}, None)


def flatten_pop_indexes(structure):
    """
    :param structure: egglib structure data

    Make a dictionary that holds indexes for each population
    """
    # make as many containers as there are pops in structure
    # empty dict
    flat_popdict = {}
    # get pops dict
    pops_dict = structure.as_dict()[0]["pops"]

    # Flatten pops_dict
    for pop in pops_dict:
        flat_popdict[pop] = [
            i for i in itt.chain.from_iterable(pops_dict[pop].values())
        ]

    # return
    return flat_popdict


def make_str_seq(window, site_pos):
    """
    :param window: an instance of VcfWindow
    :param site_pos: an integer representing a position in an alignment

    Make a sequence from window data and return a str
    """
    return "".join([site[site_pos] for site in window])


def split_win_data(window, flat_popidx):
    """
    :param window: an instance of VcfWindow
    :param flat_popidx: a dictionary of populations and the indexes of
            their individuals

    returns a dictionary with pops and their sequences
    """

    pop_win_data = {}

    for pop in flat_popidx:
        pop_win_data[pop] = [make_str_seq(window, i) for i in flat_popidx[pop]]

    return pop_win_data


def p_dist_site(base1, base2):
    """
    :param base1 base2: strings containing DNA bases

    Pairwise p-dist between two samples at a single site
    Return 1 if alleles are different, 0 if alleles are equal,
    and -1 if at least one allele is missing
    This implicitly uses pairwise deletion
    In the future, complete deletion could be used although it
    is likely too stringent
    """
    if base1 == "?" or base2 == "?":
        return -1
    elif base1 == base2:
        return 0
    else:
        return 1


def p_dist_seq(seq1, seq2):
    """
    :param seq1 seq2: strings containing DNA sequences

    Pairwise p-dist between two samples along a sliding window
    Return the p-dist, a float between 0 and 1
    """
    total_sites = 0
    cum_dist = 0
    for b1, b2 in zip(seq1, seq2):
        comp = p_dist_site(b1, b2)
        if comp >= 0:
            total_sites += 1
            cum_dist += comp
    if total_sites == 0:
        return np.nan
    else:
        return cum_dist / total_sites


def all_p_dists(window, flat_popidx, pop1, pop2):
    """
    :window: an instance of VcfWindow
    :flat_popidx: a dictionary whose keys are population names
                  and values are lists with indexes of both alleles
                  of the samples from said population
    :pop1, pop2: strings corresponding to population names

    Return a collection with all p-distances for a window
    as a numpy array
    """
    win_data = split_win_data(window, flat_popidx)

    return np.array(
        [p_dist_seq(i[0], i[1]) for i in itt.product(win_data[pop1], win_data[pop2])]
    )


def g_min(window, flat_popidx, pop1, pop2):
    """
    :window: an instance of VcfWindow
    :flat_popidx: a dictionary whose keys are population names
                  and values are lists with indexes of both alleles
                  of the samples from said population
    :pop1, pop2: strings corresponding to population names


    Return the Gmin value for a sliding window
    """
    dists = all_p_dists(window, flat_popidx, pop1, pop2)
    return np.nanmin(dists) / np.nanmean(dists)


def list_contigs(vcfpath):
    """
    :param vcfpath: a path for the vcf file

    Return the list of contigs from the vcf header
    contained in the lines starting with ##contig=<ID...
    """
    contigs = []
    with open(vcfpath, "r") as fh:
        for line in fh:
            if "##contig=<ID=" in line:
                contigs.append(line.split("=")[2].split(",")[0])
            elif "#CHROM" in line:
                break
    return contigs


def main():
    """
    main block
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf", type=str, help=("Path to the VCF file to be used."))
    parser.add_argument(
        "populations",
        type=str,
        help=(
            "Path to file containing sample ids in first "
            "column and populations they belong to in the second "
            "column."
        ),
    )
    parser.add_argument(
        "pop1",
        type=str,
        help=(
            "Name of population 1. "
            "Must be the same as the name that appears in the populations file."
        ),
    )
    parser.add_argument(
        "pop2",
        type=str,
        help=(
            "Name of population 2. "
            "Must be the same as the name that appears in the populations file."
        ),
    )
    parser.add_argument(
        "--num_sites",
        type=int,
        help=(
            "Number of variant sites that a window should have for analysis. "
            "The window bp size will be adjusted so that this number of "
            "variants is included. Default 10."
        ),
        default=10,
    )
    parser.add_argument(
        "--step_sites",
        type=int,
        help=(
            "Step size to advance a window in terms of no. of variant sites. "
            "Default 10."
        ),
        default=10,
    )
    args = parser.parse_args()

    # toggle numpy warnings for operations not involving numbers
    np.seterr(invalid="ignore")

    # parse vcf and make index
    egg.io.make_vcf_index(args.vcf)
    vcfhandle = egg.io.VcfParser(args.vcf)
    vcfhandle.load_index(args.vcf + "i")

    # parse populations file and use Structure interface to organise samples
    pop_structure = set_structure(args.populations, vcfhandle)

    flat_indexes = flatten_pop_indexes(pop_structure)

    # get a list with the names of contigs
    contigs = list_contigs(args.vcf)

    # loop over contigs and slide along each contig

    # print first line
    print(f"CHROM\tWin_Start\tWin_End\tNo_Variants\tGmin")

    # TODO: Data output needs refinement
    for contig in contigs:
        try:
            vcfhandle.goto(contig)
        except ValueError:
            continue

        slider = vcfhandle.slider(args.num_sites, args.step_sites, as_variants=True)

        while True:
            if slider.good:
                try:
                    curr_win = slider.next()
                except Exception:
                    break
                gmin_est = g_min(curr_win, flat_indexes, args.pop1, args.pop2)
                if curr_win.num_sites >= args.num_sites:
                    print(
                        f"{curr_win.chromosome}\t{curr_win.bounds[0]}\t{curr_win.bounds[1] - 1}\t{curr_win.num_sites}\t{gmin_est:.3f}"
                    )
            else:
                break


if __name__ == "__main__":
    main()
