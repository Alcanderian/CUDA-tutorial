SAMPLES=vadds

all:$(SAMPLES)

.PHONY: vadds
vadds:
	@(cd ./vadds && make && cd .. && mv ./vadds/sample ./sample_vadds) || exit 1
	