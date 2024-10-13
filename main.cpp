#include <getopt.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "addition.h"
#include "util.h"

static void print_help(const char *prog_name) {
  printf("Usage: %s [-h] [-n num_intervals]\n", prog_name);
  printf("Options:\n");
  printf("  -h : print this page.\n");
  printf("  -n : number of intervals (default: 2023)\n");
  printf("  -v  : version of reduction (default: 1)\n");

}

static int num_intervals = 2023;
static int ver = 1;

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "hn:v:")) != -1) {
    switch (c) {
      case 'n': num_intervals = atoi(optarg); break;
      case 'v': ver = atoi(optarg); break;
      case 'h':
      default: print_help(argv[0]); exit(0);
    }
  }
  printf("Options:\n");
  printf("  Number of intervals: %d\n", num_intervals);
  printf("  Version of reduction: %d\n", ver);
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  printf("Initializing... ");
  fflush(stdout);
  integral_init(num_intervals, ver);
  printf("done!\n");
  fflush(stdout);

  /* Calculate PI value */
  printf("Calculating... ");
  fflush(stdout);
  double start_time = get_current_time();
  double pi_estimate = integral(num_intervals, ver);
  double elapsed_time = get_current_time() - start_time;
  printf("done!\n");

  /* Print results */
  printf("Estimated PI value : %.16f\n", pi_estimate);
  printf("Elapsed time: %.3f sec\n", elapsed_time);
  integral_cleanup();

  return 0;
}
