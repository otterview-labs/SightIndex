FROM caddy:2
# Listen on 8080 without a privileged-port file capability. Otherwise execve can
# fail when ALL capabilities are dropped, even though no low port is needed.
RUN setcap -r /usr/bin/caddy
USER 1000:1000
