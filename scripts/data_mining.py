import os
import json
import time
from github import Github, RateLimitExceededException, GithubException
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def get_github_client():
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GitHub token not found. Please set the GITHUB_TOKEN environment variable.")
    return Github(github_token)

def get_diff(commit):
    if not commit.parents:
        return None
    
    try:
        comparison = commit.parents[0].commit.repo.compare(
            commit.parents[0].sha,
            commit.sha
        )
        return comparison.files[0].patch if comparison.files else None
    except GithubException as e:
        print(f"Could not get diff for commit {commit.sha}: {e}")
        return None


def mine_repo_commits(repo, output_file, max_commits=100):
    print(f"\nProcessing repository: {repo.full_name}")
    commits_processed = 0
    
    try:
        commits = repo.get_commits()
        
        with open(output_file, "a") as f:
            for commit in tqdm(commits[:max_commits], desc=f"Mining {repo.full_name}"):
                if commits_processed >= max_commits:
                    break

                diff = get_diff(commit)
                commit_message = commit.commit.message

                if diff and len(commit_message) > 10 and not commit.commit.message.startswith("Merge"):
                    data = {
                        "diff": diff,
                        "commit_message": commit_message
                    }
                    f.write(json.dumps(data) + "\n")
                    commits_processed += 1

    except RateLimitExceededException:
        print("Rate limit exceeded. Sleeping for 20 minutes.")
        time.sleep(20 * 60)
        mine_repo_commits(repo, output_file, max_commits)
    except GithubException as e:
        print(f"An error occurred with repository {repo.full_name}: {e}")


def main():
    g = get_github_client()
    output_file = "data/raw_data.jsonl"
    
    repositories = g.search_repositories(query="language:python", sort="stars", order="desc")
    
    print("Starting to mine GitHub repositories...")
    
    for i, repo in enumerate(repositories):
        if i >= 20:
            break
        mine_repo_commits(repo, output_file, max_commits=200)

    print(f"\nData collection complete. Data saved to {output_file}")


if __name__ == "__main__":
    main()