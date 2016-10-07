#!/bin/bash -ux

# USAGE: run with one arg: 'client' or 'server' to setup iperf3

base_args_iperf3_client="-V -c $(cat client_a/ip.txt) -J --get-server-output"

transmit_time=180
trials=1

#debug
#transmit_time=1
#trials=1

curtime=$(date --iso-8601=s | tr ':' '_')

# simple, just run
setup_iperf3_server () {
  tc qdisc list
  iperf3 -s -J
}

# deprecated in favour of -shudder- global envvar
get_cmd_iperf3_client () {
  iperf3 -V -c $(cat client_a/ip.txt) -J --get-server-output -C reno
}

# deprecated in favour of actually being flexible
setup_iperf3_client_reno () {
  tc qdisc list
  # -t : time in seconds to transmit for
  # -i : interval - pause n seconds
  # -C : linux-congestion algo 
  # -J : json
  #iperf3 -V -c $(cat client_a/ip.txt) -t 160 -i 1 -C reno
  iperf3 $base_args_iperf3_client -C reno
}

# pointless assign is pointless, this is explicitely required now
perturbation="none"
runmode=""
# arg1: client or server
# don't check whether defined, let it fail with '-u' checks
if [[ $1 == "server" ]]; then
  runmode=$1
elif [[ $1 == "client" ]]; then
  runmode=$1
fi
# require client to know exactly which perturbation being run as it is the one that collects the data
# arg2: perturbation
# must explicitly specify to avoid accidents
echo "-I-: perturbation types: 'none' 'delay' 'loss' 'corruption'"
if [[ ! -z $2 ]]; then
  perturbation=$2
fi
# arg3: algorithm
# must explicitly specify to avoid accidents
if [[ ! -z ${3:-} ]]; then
  algorithm=$3
fi

# setup args

if [[ $runmode == "server" ]]; then
  # set up perturbation conditions
  # https://wiki.linuxfoundation.org/networking/netem
  case "$perturbation" in
    none)
      rc=0
      ;;
    delay)
      # reorder
      sudo tc qdisc add dev eth1 root netem delay 75ms 10ms
      rc=$?
      ;;
    loss)
      # n are lost, successive prob dep by j on prev
      # Probn = .25 * Probn-1 + .75 * Random
      # e.g. lose 10% 
      sudo tc qdisc add dev eth1 root netem loss 10% 25%
      rc=$?
      ;;
    corruption)
      sudo tc qdisc add dev eth1 root netem corrupt 80% delay 75ms
      rc=$?
      ;;
    *)
      echo "-E-: no perturbation specified, something is wrong. use one of: 'none' 'delay' 'loss' 'corruption'";
      exit
      ;;
  esac
  if [[ $rc -ne 0 ]]; then
    exit 9
  fi
  # start server
  setup_iperf3_server
  sudo tc qdisc del dev eth1 root
  tc qdisc list
elif [[ $runmode == "client" ]]; then
  # keep track of algorithm used (reno|cubic), the perturbation, and the current time
  out_file=results_${algorithm}_${perturbation}_${curtime}.json
  # run with chosen algorithm
  for i in $(seq 1 1 $trials); do
    filenum=$(printf "%03d" $i)
    out_file=results_${algorithm}_${perturbation}_${curtime}_${filenum}.json
    #echo "iperf3 $base_args_iperf3_client -t transmit_time -C $algorithm"
    iperf3 $base_args_iperf3_client -t $transmit_time -C $algorithm >> ${out_file}
    rc=$1
    # "error":	"error - unable to connect to server: Connection refused"
    sleep 1
  done
fi

# for "removing" the date
# ls * | perl -nle 'chomp($file=$_); $file1=$file; if($file =~ s|(_2016-10-05.*?-0700)||){system "ln -s $file1 $file"}'

