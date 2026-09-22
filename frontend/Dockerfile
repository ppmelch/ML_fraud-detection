# Static frontend image: nginx serving frontend/ and reverse-proxying /api/
# to the backend service (see nginx.conf).
#
# Build context is this directory (see docker-compose.yml).
# Build:  docker build -t fraud-frontend .
# Run:    docker run -p 8080:80 fraud-frontend

FROM nginx:1.27-alpine

COPY . /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN rm -f /usr/share/nginx/html/nginx.conf

# Point app.js at the same-origin /api path proxied above instead of its
# localhost default. This only edits the copy baked into the image —
# frontend/index.html in the repo is untouched, so
# `cd frontend && python -m http.server` still works exactly as before,
# talking to http://127.0.0.1:8000 directly.
RUN sed -i \
        's#<script src="js/app.js"></script>#<script src="js/app.js" data-api-base=""></script>#' \
        /usr/share/nginx/html/index.html

EXPOSE 80
