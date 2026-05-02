#!/usr/bin/env perl
use strict;
use warnings;
use POSIX qw(log);


my $words_txt = $ARGV[0] ;

# Read word -> id mapping
my %wid;
open(my $fh, "<", $words_txt) or die "Cannot open $words_txt: $!";
while (my $line = <$fh>) {
  chomp($line);
  next if $line =~ /^\s*$/;
  my ($w, $id) = split(/\s+/, $line);
  $wid{$w} = $id;
}
close($fh);

sub w2id {
  my ($w) = @_;
  die "Word '$w' not found in $words_txt\n" if !exists $wid{$w};
  return $wid{$w};
}

sub print_layer {
  my ($from, $to, $words_ref) = @_;
  my @w = @{$words_ref};
  my $n = scalar(@w);
  die "Empty word list for layer $from->$to\n" if $n == 0;
  my $penalty = -log(1.0 / $n);

  foreach my $word (@w) {
    my $id = w2id($word);
    # FST text arc: src dst ilabel olabel weight
    print "$from $to $id $id $penalty\n";
  }
}

my $state  = 0;
my $state2 = 1;

# verb
print_layer($state, $state2, [qw(bin lay place set)]);
$state++; $state2 = $state + 1;

# colour
print_layer($state, $state2, [qw(blue green red white)]);
$state++; $state2 = $state + 1;

# prep
print_layer($state, $state2, [qw(at by in with)]);
$state++; $state2 = $state + 1;

# letter (a..v, x, y, z) -- matches your original and your words.txt
my @letters = ("a".."v", "x", "y", "z");
print_layer($state, $state2, \@letters);
$state++; $state2 = $state + 1;

# digit
print_layer($state, $state2, [qw(zero one two three four five six seven eight nine)]);
$state++; $state2 = $state + 1;

# coda
print_layer($state, $state2, [qw(again now please soon)]);

# Final state line: "final_state final_cost"
print "$state2 0.0\n";

