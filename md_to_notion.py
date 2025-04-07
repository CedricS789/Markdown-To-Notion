# Description: Converts Markdown to Notion blocks, handling nesting, removing <br>,
#              attempting newline insertion, and refining math block handling.
#              NOTE: Notion rendering of literal newlines in rich text is inconsistent.

import os
import re
from dotenv import load_dotenv
from notion_client import Client, APIResponseError
from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.dollarmath import dollarmath_plugin
import argparse
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import time
import copy
from pprint import pprint

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DEFAULT_NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID")

if not NOTION_API_KEY:
    logging.error("Error: NOTION_API_KEY not found.")
    exit(1)

# --- Notion Client Initialization ---
try:
    notion = Client(auth=NOTION_API_KEY, log_level=logging.WARNING)
    logging.info("Notion client initialized.")
except Exception as e:
    logging.error(f"Failed to initialize Notion client: {e}")
    exit(1)

# --- Markdown Parser Initialization ---
md = MarkdownIt("commonmark", {"breaks": True, "html": False})
md.enable("table")
md.use(dollarmath_plugin)
logging.info("Markdown parser initialized (breaks=True).")

# --- Helper Functions ---

def create_rich_text_object(text: str, annotations: Optional[Dict[str, Any]] = None, link_url: Optional[str] = None) -> Dict[str, Any]:
    """Creates a Notion-compatible rich text object dictionary."""
    if not isinstance(text, str):
        logging.warning(f"Non-string content for rich text: {type(text)}. Converting.")
        text = str(text)
    if len(text) > 2000:
        logging.warning(f"Rich text content exceeds 2000 characters: {len(text)}. Truncating.")
        text = text[:2000]
    obj = {"type": "text", "text": {"content": text}}
    if annotations:
        obj["annotations"] = annotations.copy()
    if link_url:
        obj["text"]["link"] = {"url": link_url}
    return obj

def create_equation_object(expression: str) -> Dict[str, Any]:
    """Creates a Notion equation object (for inline math rich text element)."""
    if not isinstance(expression, str):
        logging.warning(f"Non-string expression for equation: {type(expression)}. Converting.")
        expression = str(expression)
    return {"type": "equation", "equation": {"expression": expression.strip()}}

def create_block(block_type: str, content: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a base Notion block dictionary structure with 'children' key if applicable."""
    if not block_type or not isinstance(content, dict):
         logging.error(f"Invalid args for create_block: type='{block_type}', content type='{type(content)}'")
         return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [create_rich_text_object(f"[Error creating block: {block_type}]")]}}

    block = {"object": "block", "type": block_type, block_type: content}
    if block_type in [
        "paragraph", "bulleted_list_item", "numbered_list_item",
        "toggle", "quote", "callout", "synced_block", "template",
        "column_list", "column",
        "heading_1", "heading_2", "heading_3",
    ]:
        block["children"] = []
    return block

# --- Core Conversion Logic (V6 - Refined math_block handling) ---

def markdown_to_notion_blocks(markdown_text: str) -> List[Dict[str, Any]]:
    """
    Converts Markdown text into a hierarchical list of Notion block dictionaries.
    V6: Refines handling of math_block token state reset.
    """
    try:
        tokens = md.parse(markdown_text)
    except Exception as e:
        logging.error(f"Markdown parsing failed: {e}")
        return [{"type": "paragraph", "paragraph": {"rich_text": [create_rich_text_object(f"Error parsing Markdown: {e}")]}}]

    root_blocks = []
    parent_stack: List[Union[List[Dict[str, Any]], Dict[str, Any]]] = [root_blocks]
    list_type_stack = []
    active_annotations = {}
    link_url = None
    current_rich_text_target: Optional[List[Dict]] = None
    inside_list_item_direct_content = False

    for i, token in enumerate(tokens):
        logging.debug(f"Token[{i}]: {token.type} | Tag: {token.tag} | Level: {token.level} | Content: '{token.content[:50]}...' | Stack Depth: {len(parent_stack)} | In LI Direct: {inside_list_item_direct_content}")

        current_parent_context = parent_stack[-1]
        target_list = current_parent_context if isinstance(current_parent_context, list) else current_parent_context.setdefault("children", [])

        # --- Block Handling: Opening Tags ---
        if token.type.endswith("_open"):
            tag = token.type.replace("_open", "")
            new_block = None
            is_parent_block = False
            is_parent_list_item = isinstance(current_parent_context, dict) and "list_item" in current_parent_context.get("type", "")

            if tag == "heading":
                inside_list_item_direct_content = False
                level = int(token.tag[1])
                block_type = f"heading_{min(level, 3)}"
                new_block = create_block(block_type, {"rich_text": []})
                current_rich_text_target = new_block[block_type]["rich_text"]
                is_parent_block = True
            elif tag == "paragraph":
                if is_parent_list_item: # V3+ logic
                    logging.debug("Paragraph open inside list item: Ignoring, targeting list item rich_text.")
                    list_item_block = current_parent_context
                    list_item_type = list_item_block.get("type")
                    current_rich_text_target = list_item_block.setdefault(list_item_type, {}).setdefault("rich_text", [])
                    inside_list_item_direct_content = True
                    continue
                else:
                    inside_list_item_direct_content = False
                    new_block = create_block("paragraph", {"rich_text": []})
                    current_rich_text_target = new_block["paragraph"]["rich_text"]
                    is_parent_block = True
            elif tag == "blockquote":
                inside_list_item_direct_content = False
                new_block = create_block("quote", {"rich_text": []})
                current_rich_text_target = new_block["quote"]["rich_text"]
                is_parent_block = True
            elif tag == "bullet_list":
                list_type_stack.append("bulleted")
                inside_list_item_direct_content = False
                continue
            elif tag == "ordered_list":
                list_type_stack.append("numbered")
                inside_list_item_direct_content = False
                continue
            elif tag == "list_item":
                if list_type_stack:
                    list_type = list_type_stack[-1]
                    block_type = f"{list_type}_list_item"
                    new_block = create_block(block_type, {"rich_text": []})
                    current_rich_text_target = new_block[block_type]["rich_text"]
                    is_parent_block = True
                    inside_list_item_direct_content = True
                else:
                    logging.warning("List item token outside list context. Skipping.")
                    inside_list_item_direct_content = False
                    continue
            elif tag == "table":
                 logging.warning("Table detected, adding placeholder.")
                 inside_list_item_direct_content = False
                 new_block = create_block("paragraph", {"rich_text": [create_rich_text_object("[Table detected - content omitted]")]})
                 current_rich_text_target = None

            if new_block:
                target_list.append(new_block)
                if is_parent_block:
                    parent_stack.append(new_block)

        # --- Block Handling: Closing Tags ---
        elif token.type.endswith("_close"):
            tag = token.type.replace("_close", "")
            current_stack_top = parent_stack[-1]
            is_stack_top_dict = isinstance(current_stack_top, dict)
            block_type_at_stack_top = current_stack_top.get("type") if is_stack_top_dict else None
            pop_stack = False
            reset_target = True

            if tag == "paragraph" and inside_list_item_direct_content: # V3+ logic
                 logging.debug("Paragraph close inside list item: Ignoring.")
                 reset_target = False
                 continue

            elif tag == "list_item" and block_type_at_stack_top and "list_item" in block_type_at_stack_top:
                 pop_stack = True
                 inside_list_item_direct_content = False
            elif tag == "blockquote" and block_type_at_stack_top == "quote":
                 pop_stack = True
                 inside_list_item_direct_content = False
            elif tag == "paragraph" and block_type_at_stack_top == "paragraph":
                 pop_stack = True
                 inside_list_item_direct_content = False
            elif tag == "heading" and block_type_at_stack_top and "heading" in block_type_at_stack_top:
                 pop_stack = True
                 inside_list_item_direct_content = False

            elif tag == "bullet_list" or tag == "ordered_list":
                 if list_type_stack: list_type_stack.pop()
                 reset_target = False
            elif tag == "table":
                 pass

            if pop_stack:
                 parent_stack.pop()
                 new_parent_context = parent_stack[-1]
                 if isinstance(new_parent_context, dict):
                     parent_type = new_parent_context.get("type")
                     parent_content = new_parent_context.get(parent_type, {})
                     if "rich_text" in parent_content:
                         current_rich_text_target = parent_content["rich_text"]
                         inside_list_item_direct_content = "list_item" in parent_type if parent_type else False
                     else:
                         current_rich_text_target = None
                         inside_list_item_direct_content = False
                 else:
                     current_rich_text_target = None
                     inside_list_item_direct_content = False
            elif reset_target:
                 current_rich_text_target = None
                 if tag in ["paragraph", "heading"]:
                      inside_list_item_direct_content = False

        # --- Inline Content ---
        elif token.type == "inline":
            meaningful_children = [c for c in token.children if not ((c.type == 'text' and not c.content.strip()) or c.type == 'softbreak')]
            has_real_content = bool(meaningful_children)

            if current_rich_text_target is None and has_real_content and not inside_list_item_direct_content:
                logging.debug("Inline content requires block. Creating paragraph.")
                para_block = create_block("paragraph", {"rich_text": []})
                target_list.append(para_block)
                current_rich_text_target = para_block["paragraph"]["rich_text"]
                parent_stack.append(para_block)

            if current_rich_text_target is not None:
                for child in token.children:
                    logging.debug(f"  Inline Child: {child.type} | '{child.content[:30]}...'")

                    if child.type == "text":
                        cleaned_content = child.content.replace("<br>", "")
                        if cleaned_content and cleaned_content.strip():
                            current_rich_text_target.append(
                                create_rich_text_object(cleaned_content, active_annotations, link_url)
                            )
                        elif cleaned_content and not cleaned_content.strip() and current_rich_text_target:
                             if current_rich_text_target[-1]["type"] == "text" and not current_rich_text_target[-1]["text"]["content"].endswith(" "):
                                 current_rich_text_target[-1]["text"]["content"] += " "

                    elif child.type == "code_inline":
                         code_annotations = {"code": True, **active_annotations}
                         current_rich_text_target.append(create_rich_text_object(child.content, code_annotations, link_url))
                    elif child.type == "strong_open": active_annotations["bold"] = True
                    elif child.type == "strong_close": active_annotations.pop("bold", None)
                    elif child.type == "em_open": active_annotations["italic"] = True
                    elif child.type == "em_close": active_annotations.pop("italic", None)
                    elif child.type == "link_open": link_url = child.attrs.get('href')
                    elif child.type == "link_close": link_url = None
                    elif child.type == "softbreak":
                         if current_rich_text_target and current_rich_text_target[-1]["type"] == "text" and not current_rich_text_target[-1]["text"]["content"].endswith(" "):
                             current_rich_text_target[-1]["text"]["content"] += " "
                         logging.debug("Softbreak treated as space.")
                    elif child.type == "hardbreak":
                        logging.debug("Hardbreak found. Inserting literal newline '\\n'.")
                        if current_rich_text_target is not None:
                             current_rich_text_target.append(create_rich_text_object("\n", active_annotations, link_url))
                        else:
                             logging.warning("Hardbreak found but no rich text target. Ignoring.")
                    elif child.type == "math_inline":
                         expression = child.content.strip('$ ')
                         if expression:
                             current_rich_text_target.append(create_equation_object(expression))
                         else:
                             logging.warning(f"Empty math_inline expr: {child.content}")
                    elif child.type == "image":
                         alt_text = child.content
                         img_url = child.attrs.get('src', '')
                         if img_url:
                             logging.info(f"Detected image: {img_url}")
                             img_block = create_block("image", {
                                 "type": "external", "external": {"url": img_url},
                                 "caption": [create_rich_text_object(alt_text)] if alt_text else []
                             })
                             target_list.append(img_block)
                             current_rich_text_target = None
                             inside_list_item_direct_content = False
                         else:
                             logging.warning("Image token with no src URL. Skipping.")
            elif not has_real_content:
                 logging.debug("Ignoring inline token with no real content.")

        # --- Standalone Block Elements ---
        elif token.type == "math_block":
            # --- V6 Change: Create/append BEFORE resetting state ---
            expression = token.content.strip('$$\n ')
            new_block = None # Define new_block before potential assignment
            if expression:
                logging.debug(f"Creating block equation (type: {token.type}) expr: '{expression}'")
                new_block = create_block("equation", {"expression": expression})
                target_list.append(new_block)
                logging.debug(f"Appended equation block to target list (parent type: {type(current_parent_context).__name__})")
            else:
                logging.warning("Math block found but expression empty after stripping.")
            # Now reset state AFTER handling the block
            current_rich_text_target = None
            inside_list_item_direct_content = False

        elif token.type == "fence":
            # (Fence handling same as V5)
            current_rich_text_target = None
            inside_list_item_direct_content = False
            lang = token.info.strip().lower() or "plain text"
            lang_map = {'python': 'python', 'py': 'python', 'javascript': 'javascript', 'js': 'javascript', 'typescript': 'typescript', 'ts': 'typescript', 'html': 'html', 'css': 'css', 'bash': 'bash', 'shell': 'shell', 'json': 'json', 'yaml': 'yaml', 'markdown': 'markdown', 'sql': 'sql', 'java': 'java', 'c': 'c', 'cpp': 'c++', 'c#': 'c#', 'ruby': 'ruby', 'go': 'go', 'rust': 'rust', 'php': 'php', 'kotlin': 'kotlin', 'swift': 'swift'}
            notion_lang = lang_map.get(lang, "plain text")
            code_content = token.content
            code_rich_text = [create_rich_text_object(code_content)] if code_content else []
            new_block = create_block("code", {
                "rich_text": code_rich_text,
                "language": notion_lang
            })
            target_list.append(new_block)

        elif token.type == "hr":
            # (HR handling same as V5)
            current_rich_text_target = None
            inside_list_item_direct_content = False
            new_block = create_block("divider", {})
            target_list.append(new_block)

    # Final Debug Print
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug("--- Final Generated Block Structure ---")
        try:
            pprint(root_blocks, depth=8, width=120)
        except Exception as pp_err:
            logging.error(f"Error pretty-printing block structure: {pp_err}")
            logging.debug(f"Raw structure: {root_blocks}")
        logging.debug("-------------------------------------")

    return root_blocks


# --- Notion Upload Logic (Robust Mapping, Filtering, Retries - Same as V4/V5) ---
def append_block_tree(parent_block_id: str, blocks: List[Dict[str, Any]], depth=0):
    """
    Recursively appends blocks to Notion, handling hierarchy, batching,
    filtering empty blocks, and basic retries for API errors.
    Maps API response IDs correctly for reliable recursion.
    """
    if not blocks:
        logging.debug(f"{'  ' * depth}No blocks to append for parent {parent_block_id}.")
        return

    indent = "  " * depth
    logging.info(f"{indent}Append Tree | Parent: {parent_block_id} | Input Blocks: {len(blocks)}")

    for i in range(0, len(blocks), 100): # Process in batches of 100
        batch = blocks[i:i+100]
        # Stores info for blocks *actually sent*: (block_for_api, original_children, original_block_type)
        batch_to_send_info: List[Tuple[Dict[str, Any], Optional[List[Dict]], str]] = []

        # --- Prepare and Filter Batch ---
        for original_block_data in batch:
            block_copy = copy.deepcopy(original_block_data)
            original_children = block_copy.pop("children", None)

            block_type = block_copy.get("type")
            if not block_type or block_type not in block_copy:
                logging.warning(f"{indent}Skipping invalid block structure: {block_copy}")
                continue

            content_key = block_copy[block_type]
            is_empty = False

            # Filter blocks with effectively empty rich_text
            if isinstance(content_key, dict) and 'rich_text' in content_key:
                rich_text_content = content_key.get('rich_text', [])
                is_effectively_empty = not any(
                    (rt.get("type") == "text" and rt.get("text", {}).get("content", "").strip('\n ')) or \
                    rt.get("type") == "equation"
                    for rt in rich_text_content
                )
                allowed_empty_types = ["bulleted_list_item", "numbered_list_item", "toggle", "quote", "callout", "paragraph"]

                if is_effectively_empty and block_type not in allowed_empty_types:
                    logging.debug(f"{indent}Filtering empty content block type '{block_type}'")
                    is_empty = True
                elif is_effectively_empty:
                    content_key['rich_text'] = []
                else:
                     content_key['rich_text'] = [
                         rt for rt in rich_text_content if
                         (rt.get("type") == "text" and (rt.get("text",{}).get("content") or rt.get("text",{}).get("content") == "\n")) or \
                         rt.get("type") == "equation"
                     ]
                     if not content_key['rich_text'] and block_type not in allowed_empty_types:
                          logging.debug(f"{indent}Filtering '{block_type}' became empty after cleaning rich_text.")
                          is_empty = True

            # Filter other potentially empty blocks
            elif block_type == "image" and not content_key.get("external", {}).get("url") and not content_key.get("file", {}).get("url"):
                 logging.warning(f"{indent}Filtering image block with no URL.")
                 is_empty = True
            elif block_type == "equation" and not content_key.get("expression", "").strip():
                 logging.warning(f"{indent}Filtering equation block with empty expression.")
                 is_empty = True
            elif block_type == "code" and not content_key.get("rich_text") and content_key.get("language", "plain text") == "plain text":
                 logging.debug(f"{indent}Filtering empty code block with default language.")
                 is_empty = True

            if is_empty:
                 continue

            batch_to_send_info.append((block_copy, original_children, block_type))
        # --- End Batch Preparation ---

        if not batch_to_send_info:
            logging.info(f"{indent}Batch empty after filtering. Skipping API call.")
            continue

        prepared_batch_for_api = [info[0] for info in batch_to_send_info]

        # --- API Call with Retries ---
        max_retries = 3
        retry_delay = 1
        results = []

        for attempt in range(max_retries):
            try:
                logging.debug(f"{indent}Attempt {attempt+1}/{max_retries}: Appending batch of {len(prepared_batch_for_api)} blocks to {parent_block_id}...")
                response = notion.blocks.children.append(block_id=parent_block_id, children=prepared_batch_for_api)
                results = response.get("results", [])
                logging.info(f"{indent}API Success | Parent: {parent_block_id} | Sent: {len(prepared_batch_for_api)} | Received: {len(results)} results.")

                if len(results) != len(prepared_batch_for_api):
                    logging.error(f"{indent}API response/sent size FATAL mismatch ({len(results)} vs {len(prepared_batch_for_api)}). Aborting recursion.")
                    results = []
                    break

                # SUCCESS
                break

            except APIResponseError as e:
                logging.error(f"{indent}API Error | Attempt {attempt+1}/{max_retries} | Parent: {parent_block_id} | Status: {e.status} | Code: {e.code} | Message: {e.message}")
                failed_block_preview = prepared_batch_for_api[0] if prepared_batch_for_api else "N/A"
                logging.error(f"{indent}Failed Batch Content (first prepared block): {failed_block_preview}")
                if (e.status >= 500 or e.code == 'rate_limited') and attempt < max_retries - 1:
                    logging.warning(f"{indent}Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logging.error(f"{indent}Unrecoverable API error or max retries reached.")
                    results = []
                    break
            except Exception as e:
                logging.error(f"{indent}Unexpected Error | Attempt {attempt+1}/{max_retries} | Parent: {parent_block_id} | Error: {e}", exc_info=logging.getLogger().isEnabledFor(logging.DEBUG))
                if attempt < max_retries - 1:
                     logging.warning(f"{indent}Retrying in {retry_delay}s...")
                     time.sleep(retry_delay)
                     retry_delay *= 2
                else:
                     logging.error(f"{indent}Max retries reached for unexpected error.")
                     results = []
                     break
        # --- End API Call Loop ---

        # --- Process Results and Recurse ---
        if results:
            for idx, result in enumerate(results):
                 if idx >= len(batch_to_send_info):
                     logging.error(f"{indent}Result index {idx} out of bounds for sent info {len(batch_to_send_info)}.")
                     continue

                 newly_created_id = result.get("id")
                 (_, original_children, original_block_type) = batch_to_send_info[idx]

                 if newly_created_id and original_children:
                     logging.debug(f"{indent}  Recursion Call | New Parent: {newly_created_id} ({original_block_type}) | Children: {len(original_children)}")
                     time.sleep(0.1)
                     append_block_tree(newly_created_id, original_children, depth + 1)
                 elif newly_created_id:
                      logging.debug(f"{indent}  Leaf Node | Created: {newly_created_id} ({original_block_type}) | No children.")
                 else:
                      logging.warning(f"{indent}  Skipping recursion: Missing ID in result index {idx}. Original type: {original_block_type}")
        elif len(batch_to_send_info) > 0:
             logging.error(f"{indent}Failed to append batch to {parent_block_id} after retries/mismatch. Skipping recursion.")


# --- Main Execution Logic ---
def main():
    """Parses arguments, reads input, converts Markdown, and uploads to Notion."""
    parser = argparse.ArgumentParser(
        description="Convert Markdown (math, lists, images, etc.) to Notion blocks with nesting.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--file", metavar="MARKDOWN_FILE", help="Path to the input Markdown file.")
    input_group.add_argument("-t", "--text", metavar='"MARKDOWN_STRING"', help="Markdown text provided directly.\n(Quote strings with spaces!)")
    parser.add_argument("-p", "--page_id", default=DEFAULT_NOTION_PAGE_ID, help="Notion page/block ID to append content to.\nDefaults to NOTION_PAGE_ID from .env.")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging.")

    args = parser.parse_args()

    # Setup Logging Level
    is_debug = args.debug
    if is_debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Validate Notion Page ID
    target_page_id = args.page_id
    if not target_page_id:
        logging.error("Error: Notion Page ID is required (-p/--page_id or .env).")
        exit(1)
    logging.info(f"Target Notion Page/Block ID: {target_page_id}")

    # Load Markdown Content
    markdown_content = None
    source_description = ""
    if args.file:
        source_description = f"file '{args.file}'"
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            logging.info(f"Read Markdown from {source_description}.")
        except FileNotFoundError:
            logging.error(f"Error: File not found at '{args.file}'")
            exit(1)
        except Exception as e:
            logging.error(f"Error reading {source_description}: {e}")
            exit(1)
    elif args.text:
        source_description = "direct text argument"
        markdown_content = args.text
        logging.info(f"Using Markdown from {source_description}.")
        if not markdown_content.strip():
             logging.warning("Input markdown text is empty or whitespace only.")

    if markdown_content is None:
         logging.error("Failed to load markdown content.")
         exit(1)

    # --- Conversion and Upload ---
    logging.info(f"Converting Markdown from {source_description}...")
    root_blocks = markdown_to_notion_blocks(markdown_content)

    if root_blocks:
        logging.info(f"Conversion generated {len(root_blocks)} top-level block structure(s). Starting upload...")
        try:
            append_block_tree(target_page_id, root_blocks)
            logging.info("Upload process finished (check logs above for any API errors or warnings).")
        except Exception as e:
            logging.error(f"Upload process failed with unexpected error during initiation: {e}", exc_info=is_debug)
    else:
        logging.warning(f"Markdown conversion resulted in zero blocks. Nothing uploaded.")

    logging.info("Script finished.")


if __name__ == "__main__":
    main()