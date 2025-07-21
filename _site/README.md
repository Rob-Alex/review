# Review Site - rob-blog.co.uk

This is the review/staging environment for [rob-blog.co.uk](https://rob-blog.co.uk), accessible at [review.rob-blog.co.uk](https://review.rob-blog.co.uk).

## Purpose

- Review draft posts before publishing to main site
- Test new features and designs
- Staging environment for content approval

## Setup

This Jekyll site is automatically deployed via GitHub Pages when changes are pushed to the `main` branch.

### Local Development

```bash
bundle install
bundle exec jekyll serve
```

The site will be available at `http://localhost:4000`.

### Configuration

- **Jekyll Configuration**: `_config.yml`
- **Custom Domain**: Configured via `CNAME` file
- **Theme**: Minima with custom styling
- **Features**: Same as main site (MathJax, syntax highlighting, dark mode)

## Differences from Main Site

- Shows draft posts (`show_drafts: true`)
- Shows future-dated posts (`future: true`)
- Different site title and description
- Configured for review subdomain

## DNS Configuration Required

To enable the custom subdomain `review.rob-blog.co.uk`, add a CNAME record in your DNS settings:

```
Type: CNAME
Name: review
Target: rob-alex.github.io
```