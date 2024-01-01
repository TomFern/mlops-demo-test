cml --repo=https://github.com/TomFern/mlops-demo-test ci

cml comment create --target=commit/$SEMAPHORE_GIT_SHA report.md

---

cml comment create --target=commit/HEAD report.md