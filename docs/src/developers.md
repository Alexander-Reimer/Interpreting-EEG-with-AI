# Package Development
```@contents
Pages = ["developers.md"]
```
## For Package Developers
This section is for everybody who wants to directly contribute to this package (and for us to not forget details!).
### Documentation
To preview the documentation locally before pushing to GitHub, use `previewDocs.sh` (Linux) or manually execute

    `julia --project=docs -ie 'using BCIInterface, LiveServer; servedocs()'`
