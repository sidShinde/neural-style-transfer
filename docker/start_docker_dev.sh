IMAGE=gitlab.aptiv.today:4567/bj3pgh/dockerfiles/pydev-3p10:latest
#################################################################################
# Assemble and execute docker run command
docker_name=$(echo "$(whoami)-pydev-3p10-$(openssl rand -hex 4)")

docker_runcmd="docker run"
docker_runcmd="${docker_runcmd} --rm -it -d --gpus all"
docker_runcmd="${docker_runcmd} -v $HOME:/mnt/home"
docker_runcmd="${docker_runcmd} -v /mnt/AIArch:/mnt/AIArch"
docker_runcmd="${docker_runcmd} --name $docker_name"
docker_runcmd="${docker_runcmd} $IMAGE"

dockerid=$(eval "$docker_runcmd")
