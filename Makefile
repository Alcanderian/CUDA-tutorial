SAMPLES=vadds sgemm

all:$(SAMPLES)

.PHONY: vadds
vadds:
	@(cd ./vadds && make && cd .. && mv ./vadds/sample ./sample_vadds) || exit 1

.PHONY: sgemm
sgemm:
	@(cd ./sgemm && make && cd .. && mv ./sgemm/sample ./sample_sgemm) || exit 1
	