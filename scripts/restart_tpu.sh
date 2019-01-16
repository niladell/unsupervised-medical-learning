#!/bin/bash
# A script that restarts a specified TPU node.
# Pass the name of the TPU as argument


gcloud compute tpus stop $1 && gcloud compute tpus start $1