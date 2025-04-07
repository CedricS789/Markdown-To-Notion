# Markdown to Notion Importer

This Python script converts Markdown files into Notion pages using the official Notion API. It parses common Markdown syntax, including nested lists, math equations (inline and block), code fences, images, and more, and attempts to replicate the structure hierarchically within a target Notion page or block.

This script has evolved based on specific user feedback to handle nuances like paragraphs within list items, `\n` interpretation, and math block detection.

## Features

* **Hierarchical Structure:** Correctly handles nested lists (bulleted and ordered) by creating parent-child relationships between Notion blocks.
* **Common Markdown Blocks:**
    * Headings (`# H1` to `### H3`)
    * Paragraphs (separated by blank lines)
    * Blockquotes (`> ...`)
    * Code Fences (``` ```) with language detection (maps common names like `py`, `js` to Notion identifiers)
    * Horizontal Rules (`---`)
* **Inline Formatting:**
    * Bold (`**bold**` or `__bold__`)
    * Italic (`*italic*` or `_italic_`)
    * Inline Code (`\`code\``)
    * Links (`[text](url)`)
* **Math Support:**
    * Inline Math (`$...$`) using Notion's inline equation elements.
    * Block Math (`$$...$$`) using Notion's Equation block (Requires blank lines around it in Markdown for correct parsing).
* **Image Support:** Converts Markdown images (`![alt](url)`) with external URLs into Notion Image blocks (captioned with alt text).
* **Line Break Handling:**
    * Removes literal `<br>` tags found in the Markdown text.
    * Treats standard Markdown soft breaks (single newlines within a paragraph) as spaces.
    * Attempts to insert literal newline characters (`\n`) for Markdown hard breaks (if `breaks: True` in the parser generates `hardbreak` tokens). **Note:** Visual rendering of `\n` in Notion rich text is inconsistent.
* **API Interaction:**
    * Uses the official `notion-client` library.
    * Appends blocks in batches (up to 100) for efficiency.
    * Includes basic retries for rate limiting and server errors.
* **Debugging:** Provides a `--debug` flag for verbose logging of parsing and uploading steps.

## Limitations

* **Tables:** While the parser recognizes table syntax (`| Head | ...`), this script only inserts a simple text placeholder `[Table detected - content omitted]`. Full conversion to Notion's complex table block structure is **not implemented**.
* **Notion-Specific Blocks:** Cannot create databases, synced blocks, template buttons, columns, complex callout types/icons, etc. from standard Markdown.
* **Markdown Extensions:** Does not support non-standard syntax like GitHub Flavored Markdown task lists (`- [ ]`), footnotes, definition lists, etc., without adding specific parser plugins and conversion logic.
* **HTML:** Raw HTML tags within the Markdown source are ignored (treated as text, except for `<br>` which is removed).
* **Newline Rendering:** As noted above, Notion may not visually render the inserted `\n` characters from hard breaks as expected in standard blocks. Using blank lines in Markdown is the reliable way to create paragraph breaks.
* **Error Handling:** While basic retries exist, complex parsing errors or persistent API issues might cause the script to fail. The `--debug` log is essential for diagnosis.
* **Content Length:** Very long continuous text segments within a single rich text object might hit Notion's 2000-character limit (basic truncation is implemented, but complex splitting is not).

## Prerequisites

1.  **Python:** Python 3.7 or higher recommended.
2.  **Notion Account:** A Notion workspace where you can create integrations.
3.  **Notion Integration:**
    * Go to [My Integrations](https://www.notion.so/my-integrations).
    * Click "+ New integration".
    * Give it a name (e.g., "Markdown Importer").
    * Associate it with your desired workspace.
    * Select "Internal Integration" for capabilities. Ensure "Read content", "Insert content", and "Update content" permissions are checked.
    * Click "Submit".
    * Copy the **"Internal Integration Token"** (your `NOTION_API_KEY`). Keep this secret!
4.  **Target Notion Page:**
    * Create or navigate to the Notion page (or database page) where you want to import the Markdown content.
    * Click the "•••" menu (top right) > "Add connections" (or "Connect to") > Search for and select the integration you just created (e.g., "Markdown Importer").
    * Copy the **Page ID**. This is the 32-character hexadecimal string in the page URL (e.g., `https://www.notion.so/your-workspace/Page-Title-THIS_IS_THE_PAGE_ID?v=...`).

## Installation & Setup

1.  **Download/Clone:** Get the script file (`md_to_notion.py` or your chosen name).
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```
3.  **Install Dependencies:** Create a file named `requirements.txt` with the following content:
    ```txt
    notion-client
    python-dotenv
    markdown-it-py
    mdit-py-plugins
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
    *(Alternatively, install directly: `pip install notion-client python-dotenv markdown-it-py mdit-py-plugins`)*

4.  **Create `.env` File:** In the same directory as the script, create a file named `.env` and add your Notion API key:
    ```dotenv
    # .env file
    NOTION_API_KEY=secret_YOUR_INTERNAL_INTEGRATION_TOKEN_HERE

    # Optional: Set a default target page ID
    # NOTION_PAGE_ID=YOUR_DEFAULT_NOTION_PAGE_ID_HERE
    ```
    Replace `secret_YOUR_INTERNAL_INTEGRATION_TOKEN_HERE` with the token copied from your Notion integration settings.

## Usage

Run the script from your terminal within the activated virtual environment.

**Arguments:**

* `-f FILE`, `--file FILE`: Path to the input Markdown file. (Mutually exclusive with `-t`)
* `-t TEXT`, `--text TEXT`: Markdown text provided directly as a string. Remember to enclose in quotes if it contains spaces or special characters. (Mutually exclusive with `-f`)
* `-p PAGE_ID`, `--page_id PAGE_ID`: The ID of the Notion page or block to append the content to. Overrides `NOTION_PAGE_ID` from the `.env` file if provided. Required if not set in `.env`.
* `--debug`: Enable detailed debug logging for parsing and API calls.

**Examples:**

1.  **Import from a file:**
    ```bash
    python md_to_notion.py --file "path/to/my_notes.md" -p "YOUR_NOTION_PAGE_ID"
    ```

2.  **Import from a file using default Page ID from `.env`:**
    ```bash
    python md_to_notion.py -f "lecture_notes.md"
    ```

3.  **Import directly from text:**
    ```bash
    python md_to_notion.py -t "# Title\n\n- Item 1\n- Item 2" -p "YOUR_NOTION_PAGE_ID"
    ```

4.  **Import with Debug Logging:**
    ```bash
    python md_to_notion.py --file "complex_doc.md" -p "YOUR_NOTION_PAGE_ID" --debug
    ```

## Markdown Input Recommendations

For best results and compatibility with Notion's block structure:

* **Use Blank Lines:** Separate distinct blocks (paragraphs, headings, code fences, block equations, lists) with at least one blank line. This is standard Markdown practice and helps the parser correctly identify block boundaries.
* **Math Blocks:** Ensure `$$...$$` block equations are on their own lines and separated by blank lines from surrounding text, even within list items.
* **List Item Content:** Content intended as separate paragraphs *under* a list item should be indented and separated by blank lines from the list item's initial text.

## Troubleshooting

* **Check API Key:** Ensure the `NOTION_API_KEY` in your `.env` file is correct and doesn't include extra characters.
* **Check Page Sharing:** Verify that the target Notion page (identified by `PAGE_ID`) has been shared with your Notion integration ("Markdown Importer" or similar).
* **Check Page ID:** Double-check that the `PAGE_ID` is correct (the 32-character hex string from the URL).
* **Use `--debug`:** Run the script with the `--debug` flag. This will print:
    * Detailed parsing steps (from `markdown-it-py`).
    * The hierarchical block structure generated by `markdown_to_notion_blocks` before uploading (using `pprint`). Examine this carefully to see if the nesting and block types match your expectations.
    * Logs for the `append_block_tree` function, showing which blocks are being sent in batches and the IDs received.
    * More detailed error messages, including stack traces for unexpected errors.
* **API Errors:** Look for logged `API Error` messages, which include the status code and message from Notion. Consult the [Notion API Error Codes documentation](https://developers.notion.com/reference/errors) for more details. Common issues include invalid block structures, permission errors, or rate limits.
* **Rate Limits:** If you see `rate_limited` errors, the script has basic retries. If importing very large files frequently, you might need to add longer delays (`time.sleep`) or more sophisticated rate limit handling.

## License

MIT License, or remove if not applicable
