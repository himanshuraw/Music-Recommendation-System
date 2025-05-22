pipeline {
    agent any

    parameters {
        booleanParam(name: 'RUN_K8S', defaultValue: false, description: 'Deploy to Kubernetes?')
        booleanParam(name: 'RUN_COMPOSE', defaultValue: false, description: 'Run Docker Compose?')
    }

    stages {
        stage('Build & Push Docker Images') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'docker-hub-credentials',
                    usernameVariable: 'DOCKERHUB_USER',
                    passwordVariable: 'DOCKERHUB_PASS'
                )]) {
                    sh '''
                        echo "$DOCKERHUB_PASS" | docker login -u "$DOCKERHUB_USER" --password-stdin

                        echo "Building Docker images..."
                        docker build -t $DOCKERHUB_USER/backend:latest ./backend
                        docker build -t $DOCKERHUB_USER/frontend:latest ./frontend
                        docker build -t $DOCKERHUB_USER/ml_service:latest ./ml_service

                        echo "Pushing images to Docker Hub..."
                        docker push $DOCKERHUB_USER/backend:latest
                        docker push $DOCKERHUB_USER/frontend:latest
                        docker push $DOCKERHUB_USER/ml_service:latest

                        docker logout
                    '''
                }
            }
        }

        stage('Run Ansible Playbook') {
            steps {
                withCredentials([
                    sshUserPrivateKey(credentialsId: 'ssh-key', keyFileVariable: 'SSH_KEY'),
                    string(credentialsId: 'ansible-vault-password', variable: 'VAULT_PASS')
                ]) {
                    sh '''
                        echo "Running Ansible Playbook..."
                        export ANSIBLE_HOST_KEY_CHECKING=False

                        VAULT_PASS_FILE=$(mktemp)
                        echo "$VAULT_PASS" > "$VAULT_PASS_FILE"
                        chmod 600 "$VAULT_PASS_FILE"

                        ansible-playbook -i ansible/inventory.ini ansible/playbook.yml \
                            --private-key=$SSH_KEY --vault-password-file="$VAULT_PASS_FILE"

                        rm -f "$VAULT_PASS_FILE"
                    '''

                    script {
                        if (params.RUN_COMPOSE) {
                            echo "RUN_COMPOSE is true, running docker-compose up"
                            sh '''
                                echo "Starting Docker Compose stack..."
                                docker-compose -f docker-compose.yml up -d
                            '''
                        } else {
                            echo "RUN_COMPOSE is false, skipping docker-compose"
                        }
                    }
                }
            }
        }

        stage('Deploy to Kubernetes') {
            when {
                expression { return params.RUN_K8S }
            }
            steps {
                withCredentials([file(credentialsId: 'kubeconfig-secret-file', variable: 'KUBECONFIG_FILE')]) {
                    sh '''
                        echo "Deploying to Kubernetes..."
                        export KUBECONFIG=$KUBECONFIG_FILE

                        kubectl apply -f k8s/backend.yml
                        kubectl apply -f k8s/frontend.yml
                        kubectl apply -f k8s/ml_service.yml
                        kubectl apply -f k8s/elasticsearch.yml
                        kubectl apply -f k8s/kibana.yml
                        kubectl apply -f k8s/logstash.yml
                        kubectl apply -f k8s/pv.yaml
                    '''
                }
            }
        }
    }

    post {
        always {
            echo 'Pipeline complete.'
        }
    }
}
