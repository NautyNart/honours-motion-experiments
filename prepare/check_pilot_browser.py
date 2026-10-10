"""Optional development check: headless installed Chrome/Edge, no browser download."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('viewer', type=Path)
    parser.add_argument('--browser', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=str(args.browser), headless=True)
        page = browser.new_page(viewport={'width': 1200, 'height': 900}, accept_downloads=True)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.viewer.resolve().as_uri())
        count = page.locator('article').count()
        assert count > 0, 'No review cards rendered'
        page.locator('input[type=range]').first.fill('5')
        page.locator('input[type=range]').first.dispatch_event('input')
        page.screenshot(path=str(args.output / 'review.png'))
        with page.expect_download() as event:
            page.locator('#save').click()
        path = args.output / 'draft_browser_download.json'
        event.value.save_as(path)
        draft = json.loads(path.read_text())
        assert len(draft['entries']) == count
        assert not errors, errors
        result = {'cards': count, 'javascript_errors': errors, 'download_round_trip': True,
                  'note': 'Saved draft only; no human approval or outcome was supplied.'}
        (args.output / 'browser_check.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        browser.close()

if __name__ == '__main__':
    main()
