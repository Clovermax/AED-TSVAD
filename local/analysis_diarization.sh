#!/usr/bin/env bash
#

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

score_area=
collar=$1

. "${SCRIPT_DIR}/../utils/parse_options.sh"
exp_name=$2
system=$3
type=$4
ref_rttm_path=$5
hyp_rttm_path=$6

uem=$7
# uem=None

#tempdir=$( mktemp  -d  /tmp/eval_diarization.XXXXXX )

echo $SCRIPT_DIR

tempdir=$exp_name/$system/$type/
echo $tempdir
mkdir -p $tempdir

# "${SCRIPT_DIR}/md-eval-22.pl" $score_area -c $collar -afc -r $ref_rttm_path -s $hyp_rttm_path > ${tempdir}/temp.info

if [ -f $uem ];then
    echo uem
    ${SCRIPT_DIR}/md-eval-22.pl $score_area -u $uem -c $collar -afc -r $ref_rttm_path -s $hyp_rttm_path > ${tempdir}/temp.info
else
    ${SCRIPT_DIR}/md-eval-22.pl $score_area -c $collar -afc -r $ref_rttm_path -s $hyp_rttm_path > ${tempdir}/temp.info
fi
grep SCORED ${tempdir}/temp.info | cut -d "=" -f 2 | cut -d " " -f 1 > ${tempdir}/SCORED.list
grep MISSED ${tempdir}/temp.info | cut -d "=" -f 2 | cut -d " " -f 1 > ${tempdir}/MISSED.list
grep FALARM ${tempdir}/temp.info | cut -d "=" -f 2 | cut -d " " -f 1 > ${tempdir}/FALARM.list
grep "SPEAKER ERROR" ${tempdir}/temp.info | cut -d "=" -f 2 | cut -d " " -f 1 > ${tempdir}/SPEAKER.list
grep OVERALL ${tempdir}/temp.info | cut -d "=" -f 4 | cut -d ")" -f 1 > ${tempdir}/session.list
sed -i '$d' ${tempdir}/session.list
echo "ALL" >> ${tempdir}/session.list

# for l in `cat ${tempdir}/session.list`;do
#     grep $l $ref_rttm_path | awk '{print $8}' | sort | uniq | wc -l
# done > ${tempdir}/oracle_spknum.list

# for l in `cat ${tempdir}/session.list`;do
#     grep $l $hyp_rttm_path | awk '{print $8}' | sort | uniq | wc -l
# done > ${tempdir}/diarized_spknum.list

# 提取 oracle 中每个 session 的说话人数量
while read l; do
    awk -v sid="$l" '$2 == sid {print $8}' "$ref_rttm_path" | sort | uniq | wc -l
done < "${tempdir}/session.list" > "${tempdir}/oracle_spknum.list"

# 提取 diarized 中每个 session 的说话人数量
while read l; do
    awk -v sid="$l" '$2 == sid {print $8}' "$hyp_rttm_path" | sort | uniq | wc -l
done < "${tempdir}/session.list" > "${tempdir}/diarized_spknum.list"

paste -d " " ${tempdir}/session.list ${tempdir}/SCORED.list ${tempdir}/MISSED.list \
             ${tempdir}/FALARM.list ${tempdir}/SPEAKER.list ${tempdir}/oracle_spknum.list \
             ${tempdir}/diarized_spknum.list > ${tempdir}/temp.details

awk '{printf "%s %.2f %.2f %.2f %.2f %d %d\n",$1,$4/$2*100,$3/$2*100,$5/$2*100,($3+$4+$5)/$2*100,$6,$7}' ${tempdir}/temp.details > ${tempdir}/temp.info1
echo "session FA MISS SPKERR DER ORACLE_SPKNUM DIARIZED_SPKNUM" > ${tempdir}/temp.details
grep -v "ALL" ${tempdir}/temp.info1 | sort -n -k 5 >> ${tempdir}/temp.details
grep "ALL" ${tempdir}/temp.info1 >> ${tempdir}/temp.details

column -t ${tempdir}/temp.details

# rm -rf ${tempdir}