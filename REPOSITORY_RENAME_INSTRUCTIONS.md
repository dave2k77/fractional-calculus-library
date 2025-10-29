# Repository Rename Instructions

## Overview

To rename the GitHub repository from `fractional-calculus-library` to `hpfracc` to match the actual library name, follow these steps:

## Step 1: Rename Repository on GitHub ✅ COMPLETED

~~1. Go to your repository on GitHub: https://github.com/dave2k77/fractional-calculus-library~~
~~2. Click on **Settings** (in the repository navigation bar)~~
~~3. Scroll down to the **Danger Zone** section at the bottom~~
~~4. Click **Change repository name**~~
~~5. Enter the new name: `hpfracc`~~
~~6. Click **I understand, change repository name**~~

**Status:** ✅ Repository renamed to `hpfracc` on GitHub

**Important Notes:**
- GitHub automatically redirects the old URL to the new one
- All existing references will continue to work temporarily
- GitHub Pages, if enabled, may need to be reconfigured
- ReadTheDocs integration may need repository name update

## Step 2: Update Local Git Remote ✅ COMPLETED

~~After renaming on GitHub, update your local repository:~~

```bash
git remote set-url origin https://github.com/dave2k77/hpfracc.git
git remote -v  # Verify the change
```

**Status:** ✅ Local git remote updated to `https://github.com/dave2k77/hpfracc.git`

## Step 3: Update External Services

### ReadTheDocs
- Go to https://readthedocs.org/dashboard/
- Find your project (currently `hpfracc` or `fractional-calculus-library`)
- Go to **Settings** → **Integrations**
- Update the **GitHub repository** field if needed
- The documentation URL should remain `hpfracc.readthedocs.io`

### Zenodo Integration
- Go to https://zenodo.org/account/settings/github/
- Find the repository in your connected repositories
- Verify the integration still works (it should automatically update)
- Future releases will use the new repository name

### PyPI Trusted Publishing
- Go to https://pypi.org/manage/account/publishing/
- Find the pending publisher for `hpfracc`
- Update the **Repository name** field from `fractional_calculus_library` to `hpfracc`
- Verify the workflow file name matches

### CI/CD Workflows
- The workflows in `.github/workflows/` should continue to work
- Verify they run successfully after the rename
- No changes needed if workflows use relative paths

## Step 4: Verify Everything Works

1. **Push/Pull**: Test that `git push` and `git pull` work correctly
2. **GitHub Issues**: Verify issue URLs still work (they should redirect)
3. **Documentation**: Check that ReadTheDocs builds successfully
4. **Badges**: Verify that badges in README.md still work
5. **PyPI Publishing**: Test that the next release publishes correctly

## Step 5: Update Bookmarked Links

If you have bookmarked:
- Old: `https://github.com/dave2k77/fractional-calculus-library`
- New: `https://github.com/dave2k77/hpfracc`

## Notes

- All code references have been updated in this commit
- The PyPI package name (`hpfracc`) remains unchanged
- Documentation URLs may need verification
- Existing clones will need to update their remotes manually
- GitLab mirrors (if any) will need separate updates

## Benefits

✅ Consistent naming: repository name matches library name (`hpfracc`)
✅ Cleaner URLs: `github.com/dave2k77/hpfracc`
✅ Better discoverability: repository name matches PyPI package
✅ Professional appearance: consistent branding across platforms

