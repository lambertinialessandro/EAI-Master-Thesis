

FROM bitnami/pytorch

MAINTAINER alessandro lambertini <lambertini.1938390@studenti.uniroma1.it>

# executed when build
# 	docker build -t DeepVO:1.0 .
RUN mkdir -p /home/app

COPY . /home/app

# execute when run
#	docker run dockerID
CMD ["echo", "Hello World...!"]

