# Deploying to AWS (container on App Runner)

These steps were **not executed from the development sandbox** (no Docker or
AWS access there). Treat them as a checklist and fix details as you go.

## 1. Build and test the image locally
```bash
docker build -t phishing-url-detector .
docker run --rm -p 8000:8000 phishing-url-detector
# open http://localhost:8000  and  http://localhost:8000/api/health
```
Optional LLM explanations: add `-e ANTHROPIC_API_KEY=...` to `docker run`.

## 2. Push to Amazon ECR
```bash
AWS_REGION=ap-south-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/phishing-url-detector

aws ecr create-repository --repository-name phishing-url-detector --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
docker tag phishing-url-detector:latest $REPO:latest
docker push $REPO:latest
```

## 3. Run it
Simplest option: **AWS App Runner** -> Create service -> Source: Amazon ECR
image above -> Port `8000` -> health check path `/api/health`. Give it at
least 2 GB memory (TensorFlow). Add `ANTHROPIC_API_KEY` as a secret
(Secrets Manager) if you want LLM explanations.

Alternatives: ECS Fargate (more control) or a single EC2 instance running
`docker run`.

## Known limitation
The prediction history is a SQLite file inside the container, so it resets
whenever the service redeploys. For persistence, move `PredictionStore`
to Amazon RDS (PostgreSQL) or mount EFS.
