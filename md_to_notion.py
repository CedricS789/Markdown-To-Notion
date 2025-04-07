# Description: Converts Markdown to Notion blocks, handling nesting, removing <br>,
#              attempting newline insertion, and refining math block handling.
#              Provides enhanced logging for common Notion API errors.
#              Logs output to both console and md_to_notion.log file.
#              NOTE: Notion rendering of literal newlines ('\n') in rich text is inconsistent.

# --- Imports ---
import os
import re
from dotenv import load_dotenv # For loading environment variables from .env file
from notion_client import Client, APIResponseError # Official Notion SDK
from markdown_it import MarkdownIt # Markdown parser library
from markdown_it.token import Token # Type hint for tokens
from mdit_py_plugins.dollarmath import dollarmath_plugin # Plugin for $...$ and $$...$$ math
import argparse # For command-line argument parsing
import logging # For logging information and errors
import logging.handlers # For file logging handler
from typing import List, Dict, Any, Optional, Union, Tuple # Type hinting
import time # For sleep/retry delays
import copy # For deep copying block structures
from pprint import pprint # For pretty-printing debug output
import sys # For exiting script cleanly on critical errors

# --- Configuration ---
# Load environment variables from a .env file if it exists
load_dotenv()

# --- Logging Setup ---
# Configure logging to output to both a file and the console.
LOG_FILENAME = "md_to_notion.log"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S' # Log message timestamp format

# Get root logger instance
logger = logging.getLogger()
# Set root logger level to DEBUG. This allows all messages (DEBUG and above)
# to be processed by the logger. Handlers below will filter based on their own levels.
logger.setLevel(logging.DEBUG)

# Create a formatter to define the log message structure
formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

# --- File Handler ---
# Logs all messages (DEBUG and above) to the specified file.
try:
    # Use 'a' for append mode (keeps logs across runs). Use 'w' to overwrite each run.
    file_handler = logging.FileHandler(LOG_FILENAME, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.DEBUG) # Ensure file handler captures all DEBUG level messages
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Initial confirmation message printed directly to console during setup
    print(f"INFO - Logging detailed output (DEBUG level) to file: {os.path.abspath(LOG_FILENAME)}")
except Exception as e:
    # Fallback if the log file cannot be opened (e.g., permissions error)
    print(f"ERROR - Failed to set up log file '{LOG_FILENAME}': {e}", file=sys.stderr)
    print("ERROR - Logging will proceed to console only.", file=sys.stderr)

# --- Console Handler ---
# Logs messages to the console (stderr). Level is set dynamically later based on --debug flag.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Default to INFO level for console
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# --- End Logging Setup ---


# --- Environment Variables ---
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DEFAULT_NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID") # Default target page if not specified via args

# --- Notion API Limits (Constants defined in code for clarity and stability) ---
# These reflect documented limits of the Notion API as of the time of writing.
# !!!!! DO NOT change these unless the Notion API documentation indicates a change !!!!!
NOTION_MAX_RICH_TEXT_LENGTH = 2000
NOTION_MAX_EQUATION_EXPR_LENGTH = 1000
NOTION_MAX_CHILDREN_PER_APPEND = 100

# Block types known to support children array for nesting content
# Based on Notion API documentation. Used in create_block and markdown_to_notion_blocks.
NESTABLE_BLOCK_TYPES = {
    "paragraph", "bulleted_list_item", "numbered_list_item",
    "toggle", "quote", "callout", "synced_block", "template",
    "column", "child_page", "child_database", # Note: column_list is created implicitly by Notion
    "heading_1", "heading_2", "heading_3", "table_row" # table_row children are cells (which are block objects)
}

# --- Pre-flight Checks ---
# Ensure the Notion API key is available before proceeding.
if not NOTION_API_KEY:
    logging.critical("CRITICAL ERROR: NOTION_API_KEY not found in environment variables or .env file.")
    logging.critical("Please ensure your Notion API key is correctly set as an environment variable or in a '.env' file.")
    sys.exit(1) # Exit cleanly if key is missing

# --- Notion Client Initialization ---
try:
    # Determine initial log level for the Notion client library itself.
    # It can be verbose, so reduce its output unless the script's debug flag is active.
    # --- MODIFICATION: Commented out dynamic level setting based on --debug to prevent notion_client DEBUG logs ---
    # notion_client_log_level = logging.DEBUG if '--debug' in sys.argv else logging.WARNING
    notion_client_log_level = logging.WARNING # Set a fixed level (e.g., WARNING)
    notion = Client(auth=NOTION_API_KEY, log_level=notion_client_log_level)
    logging.info("Notion client initialized.")
except Exception as e:
    # Catch potential errors during client initialization (e.g., invalid key format, network issues)
    logging.critical(f"CRITICAL ERROR: Failed to initialize Notion client: {e}")
    logging.critical("This might be due to an invalid API key format or initial network connectivity issues.")
    sys.exit(1)

# --- Markdown Parser Initialization ---
# Use CommonMark specification as the base for parsing.
# breaks=True: Treat single newlines in Markdown source as hard line breaks.
# html=False: Disable parsing of raw HTML tags in Markdown for security and predictability.
md = MarkdownIt("commonmark", {"breaks": True, "html": False})
md.enable("table") # Explicitly enable GitHub Flavored Markdown style tables
md.use(dollarmath_plugin) # Enable parsing of $inline$ and $$block$$ math syntax
logging.info("Markdown parser initialized (commonmark, breaks=True, table, dollarmath).")

# --- Helper Functions ---

def create_rich_text_object(text: str, annotations: Optional[Dict[str, Any]] = None, link_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Creates a Notion-compatible rich text object dictionary for text content.

    Handles type checking, enforces Notion's 2000-character limit by truncating,
    applies provided annotations (styling), and adds link objects.

    Args:
        text: The string content for the rich text object.
        annotations: Optional dictionary defining text styling (bold, italic, etc.).
        link_url: Optional URL string to create a hyperlink for this text.

    Returns:
        A dictionary formatted as a Notion rich text object.

    Ref: https://developers.notion.com/reference/rich-text
    """
    # Ensure input text is actually a string.
    if not isinstance(text, str):
        logging.warning(f"Non-string content received for rich text: {type(text)}. Attempting string conversion.")
        text = str(text)

    # Enforce Notion API limit for the length of a single rich text object.
    # Important: This limit applies *per object*, not per block's entire rich_text array.
    if len(text) > NOTION_MAX_RICH_TEXT_LENGTH:
        logging.warning(f"Rich text content exceeds Notion API limit of {NOTION_MAX_RICH_TEXT_LENGTH} characters ({len(text)}). Truncating.")
        text = text[:NOTION_MAX_RICH_TEXT_LENGTH]

    # Base structure for a Notion text object.
    obj: Dict[str, Any] = {"type": "text", "text": {"content": text}}

    # Apply annotations if provided (e.g., {"bold": True, "color": "red"})
    if annotations:
        obj["annotations"] = annotations.copy() # Use a copy to prevent unintended modifications

    # Add a link object if a URL is provided.
    if link_url:
        # Basic sanity check for common URL protocols. Allows relative links starting with /.
        if not re.match(r'^(https?|mailto|tel|ftp|file|/):', link_url, re.IGNORECASE):
            logging.warning(f"Potentially invalid link URL format detected: '{link_url}'. URL should ideally include a scheme (e.g., 'https://') or be a relative path starting with '/'. Using provided URL anyway.")
        obj["text"]["link"] = {"url": link_url}

    return obj

def create_equation_object(expression: str) -> Dict[str, Any]:
    """
    Creates a Notion equation object (for inline math rich text or block content).

    Handles type checking, strips whitespace, enforces Notion's 1000-character limit
    for expressions by truncating, and handles empty expressions.

    Args:
        expression: The LaTeX string for the equation.

    Returns:
        A dictionary formatted as a Notion equation object OR
        a placeholder rich text object if the expression is empty.

    Ref: https://developers.notion.com/reference/rich-text#equation-objects
    Ref: https://developers.notion.com/reference/block#equation-blocks
    """
    # Ensure input is a string.
    if not isinstance(expression, str):
        logging.warning(f"Non-string expression received for equation: {type(expression)}. Attempting string conversion.")
        expression = str(expression)

    # Remove leading/trailing whitespace often present in $$...$$ blocks.
    expr_stripped = expression.strip()

    # Enforce Notion API limit for the length of an equation expression string.
    if len(expr_stripped) > NOTION_MAX_EQUATION_EXPR_LENGTH:
         logging.warning(f"Equation expression exceeds Notion API limit of {NOTION_MAX_EQUATION_EXPR_LENGTH} characters ({len(expr_stripped)}). Truncating.")
         expr_stripped = expr_stripped[:NOTION_MAX_EQUATION_EXPR_LENGTH]

    # Prevent creating invalid equation objects with empty expressions.
    if not expr_stripped:
        logging.warning(f"Cannot create equation object with empty expression. Original input: '{expression}'. Returning placeholder text.")
        # Return a simple text object indicating the issue instead of an invalid equation object.
        return create_rich_text_object("[Empty Math Expression]")

    # Standard structure for a Notion equation object (used within rich_text arrays for inline math).
    # For block-level equations, this dict gets nested under the 'equation' key of the main block.
    return {"type": "equation", "equation": {"expression": expr_stripped}}

def create_block(block_type: str, content_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a base Notion block dictionary structure ready for the API.

    Automatically adds an empty 'children' list to the content dictionary
    if the specified block_type supports nesting according to NESTABLE_BLOCK_TYPES.
    Includes basic validation for input arguments.

    Args:
        block_type: The string representing the Notion block type (e.g., "paragraph", "heading_1").
        content_dict: A dictionary containing the specific content for that block type
                      (e.g., {"rich_text": [...]}, {"expression": "..."}).

    Returns:
        A dictionary formatted as a Notion block object.

    Ref: https://developers.notion.com/reference/block
    """
    # Validate essential inputs.
    if not block_type or not isinstance(content_dict, dict):
         logging.error(f"Invalid arguments provided to create_block: Type='{block_type}', Content Type='{type(content_dict)}'")
         # Return a distinct paragraph block indicating a script error occurred.
         return {
             "object": "block",
             "type": "paragraph",
             "paragraph": {
                 "rich_text": [create_rich_text_object(f"[Script Error: Failed to create block of type '{block_type}']")]
             }
         }

    # Standard structure for all Notion block objects.
    block = {"object": "block", "type": block_type, block_type: content_dict}

    # Add an empty 'children' list *within the content dict* (e.g., block['paragraph']['children'])
    # for block types that support nesting, if 'children' isn't already present.
    # This simplifies the recursive upload logic in `append_block_tree`.
    if block_type in NESTABLE_BLOCK_TYPES:
        if "children" not in content_dict:
             block[block_type]["children"] = []

    return block

# --- Core Conversion Logic ---

def markdown_to_notion_blocks(markdown_text: str) -> List[Dict[str, Any]]:
    """
    Converts a Markdown string into a hierarchical list of Notion block dictionaries.

    Iterates through tokens generated by the MarkdownIt parser and maps them to
    corresponding Notion block structures, handling nesting and inline formatting.

    Args:
        markdown_text: The Markdown string to convert.

    Returns:
        A list of dictionaries, where each dictionary represents a top-level Notion block
        (potentially containing nested 'children' blocks). Returns an error block list
        if parsing fails.
    """
    try:
        # Parse the Markdown text into a stream of tokens using MarkdownIt.
        tokens = md.parse(markdown_text)
    except Exception as e:
        logging.error(f"Markdown parsing failed unexpectedly: {e}", exc_info=logging.getLogger().isEnabledFor(logging.DEBUG))
        # Return a single error block to indicate the parsing failure.
        return [create_block("paragraph", {"rich_text": [create_rich_text_object(f"[Script Error: Failed to parse Markdown - {e}]")]})]

    # --- State Variables for Conversion ---
    root_blocks: List[Dict[str, Any]] = [] # The final list of top-level blocks.
    # parent_stack manages nesting. It holds references to block dictionaries or the root_blocks list.
    # New blocks are appended to the 'children' list of the block dict on top of the stack,
    # or to root_blocks if the stack only contains the root list.
    parent_stack: List[Union[List[Dict[str, Any]], Dict[str, Any]]] = [root_blocks]
    list_type_stack: List[str] = [] # Tracks 'bulleted' or 'numbered' for list items.
    active_annotations: Dict[str, Any] = {} # Tracks current inline styles (bold, italic, code, etc.).
    link_url: Optional[str] = None # Tracks the URL for the currently open link tag.
    # current_rich_text_target points to the specific 'rich_text' list within a block's
    # content where inline elements (text, equations, mentions) should be added.
    current_rich_text_target: Optional[List[Dict]] = None
    # Special flag for list items: Markdown often puts simple text directly in <li><p>text</p></li>.
    # In Notion, this maps to the list item's *own* rich_text, not a nested paragraph block.
    # This flag helps direct inline content correctly in that specific scenario.
    inside_list_item_direct_content: bool = False

    # --- Token Processing Loop ---
    for i, token in enumerate(tokens):
        # Debug log for tracking token processing flow.
        logging.debug(f"Token[{i}]: {token.type} | Tag: {token.tag} | Level: {token.level} | Content: '{token.content[:50]}...' | Stack Depth: {len(parent_stack)} | In LI Direct: {inside_list_item_direct_content}")

        # --- Determine Target List ---
        # Find where the current block should be appended based on the parent stack context.
        current_parent_context = parent_stack[-1]
        target_list: List[Dict]
        if isinstance(current_parent_context, list):
            # If the top of the stack is a list (either root_blocks or a 'children' list).
            target_list = current_parent_context
        elif isinstance(current_parent_context, dict):
            # If the top is a block dictionary, get its 'children' list.
            parent_type = current_parent_context.get("type")
            if parent_type and parent_type in current_parent_context:
                block_content = current_parent_context.setdefault(parent_type, {})
                # Ensure the children list exists (should be guaranteed by create_block)
                target_list = block_content.setdefault("children", [])
            else:
                # Fallback if the block structure on the stack is invalid.
                logging.error(f"Invalid parent block context on stack (missing type key or content): {current_parent_context}. Appending to root.")
                target_list = root_blocks
        else:
            # Fallback for unexpected stack state.
            logging.error(f"Unexpected parent context type on stack: {type(current_parent_context)}. Appending to root.")
            target_list = root_blocks

        # --- Block Type Handling ---

        # --- 1. Opening Tags (*_open tokens) ---
        if token.type.endswith("_open"):
            tag = token.type.replace("_open", "") # e.g., 'heading', 'paragraph', 'list_item'
            new_block: Optional[Dict[str, Any]] = None # Initialize potential new block
            is_parent_block = False # Does this block become a context for children?
            is_parent_list_item = isinstance(current_parent_context, dict) and "list_item" in current_parent_context.get("type", "")

            # --- Map Markdown opening tags to Notion block types ---
            if tag == "heading":
                inside_list_item_direct_content = False # Headings reset this flag
                level = int(token.tag[1]) # h1, h2, etc.
                block_type = f"heading_{min(level, 3)}" # Map to heading_1, heading_2, or heading_3
                new_block = create_block(block_type, {"rich_text": []})
                current_rich_text_target = new_block[block_type]["rich_text"]
                is_parent_block = True # Subsequent inline content belongs to this heading
            elif tag == "paragraph":
                if is_parent_list_item:
                    # Paragraphs immediately inside list items map to the list item's rich_text
                    logging.debug("Paragraph open inside list item: Targeting list item's rich_text.")
                    list_item_block = current_parent_context
                    list_item_type = list_item_block.get("type", "bulleted_list_item")
                    list_content = list_item_block.setdefault(list_item_type, {})
                    current_rich_text_target = list_content.setdefault("rich_text", [])
                    inside_list_item_direct_content = True
                    continue # Don't create a separate paragraph block
                else:
                    # Regular paragraph
                    inside_list_item_direct_content = False
                    new_block = create_block("paragraph", {"rich_text": []})
                    current_rich_text_target = new_block["paragraph"]["rich_text"]
                    is_parent_block = True # Paragraphs can contain children (e.g., nested lists)
            elif tag == "blockquote":
                inside_list_item_direct_content = False
                new_block = create_block("quote", {"rich_text": []})
                current_rich_text_target = new_block["quote"]["rich_text"]
                is_parent_block = True # Blockquotes can contain other blocks
            elif tag == "bullet_list":
                list_type_stack.append("bulleted") # Track that we are inside a bulleted list
                inside_list_item_direct_content = False
                continue # The list itself doesn't create a block, its items do
            elif tag == "ordered_list":
                list_type_stack.append("numbered") # Track that we are inside a numbered list
                inside_list_item_direct_content = False
                continue # The list itself doesn't create a block
            elif tag == "list_item":
                if list_type_stack: # Ensure we are actually inside a list context
                    list_type = list_type_stack[-1] # Get the type ('bulleted' or 'numbered')
                    block_type = f"{list_type}_list_item"
                    new_block = create_block(block_type, {"rich_text": []})
                    current_rich_text_target = new_block[block_type]["rich_text"] # Subsequent inline content goes here first
                    is_parent_block = True # List items are parents for nested content
                    inside_list_item_direct_content = True # Mark that inline content belongs directly to the item
                else:
                    # This indicates malformed Markdown or a parser issue
                    logging.warning("List item token encountered outside of a list context. Skipping.")
                    inside_list_item_direct_content = False
                    continue
            elif tag == "table":
                 # Basic table handling: create a placeholder paragraph.
                 # Full table support would require parsing thead, tbody, tr, th, td tokens
                 # and constructing Notion table/table_row blocks with specific cell structures.
                 logging.warning("Table detected, adding a placeholder paragraph. Full table conversion not supported.")
                 inside_list_item_direct_content = False
                 new_block = create_block("paragraph", {"rich_text": [create_rich_text_object("[Table detected - Content omitted]")]})
                 current_rich_text_target = None # Placeholder doesn't accept further rich text

            # --- Append and Update Stack ---
            if new_block:
                target_list.append(new_block)
                # If the new block can act as a parent for subsequent blocks (e.g., list item, paragraph),
                # push it onto the stack to define the current nesting context.
                if is_parent_block:
                    parent_stack.append(new_block)

        # --- 2. Closing Tags (*_close tokens) ---
        elif token.type.endswith("_close"):
            tag = token.type.replace("_close", "")
            current_stack_top = parent_stack[-1] if parent_stack else None
            is_stack_top_dict = isinstance(current_stack_top, dict)
            block_type_at_stack_top = current_stack_top.get("type") if is_stack_top_dict else None
            pop_stack = False # Should we pop the parent context?
            reset_target = True # Should the rich text target be cleared?

            # Ignore paragraph closes immediately inside list items, as they don't correspond to Notion blocks.
            if tag == "paragraph" and inside_list_item_direct_content:
                 logging.debug("Paragraph close inside list item: Ignoring.")
                 reset_target = False # Keep targeting the list item's rich text
                 continue

            # Check if the closing tag matches the block type on top of the stack
            # If it matches, we are closing that block's context and should pop the stack.
            if tag == "list_item" and block_type_at_stack_top and "list_item" in block_type_at_stack_top:
                 pop_stack = True
                 inside_list_item_direct_content = False # Exiting direct content mode
            elif tag == "blockquote" and block_type_at_stack_top == "quote":
                 pop_stack = True
                 inside_list_item_direct_content = False
            elif tag == "paragraph" and block_type_at_stack_top == "paragraph":
                 pop_stack = True
                 inside_list_item_direct_content = False
            elif tag == "heading" and block_type_at_stack_top and "heading" in block_type_at_stack_top:
                 pop_stack = True
                 inside_list_item_direct_content = False
            # List closes only pop from the list_type_stack, not the block parent_stack.
            elif tag == "bullet_list" or tag == "ordered_list":
                 if list_type_stack: list_type_stack.pop()
                 reset_target = False # No block context to pop or target to reset
            elif tag == "table":
                 reset_target = False # Placeholder wasn't pushed onto stack

            # --- Pop Stack and Reset Target ---
            if pop_stack and len(parent_stack) > 1: # Avoid popping the root list itself
                 parent_stack.pop()
                 # Update the rich text target based on the *new* parent context we returned to.
                 new_parent_context = parent_stack[-1]
                 if isinstance(new_parent_context, dict):
                     parent_type = new_parent_context.get("type")
                     # Check if this parent type supports rich text directly.
                     if parent_type in ["paragraph", "heading_1", "heading_2", "heading_3",
                                        "bulleted_list_item", "numbered_list_item",
                                        "quote", "callout", "toggle"]:
                        # Ensure content dict and rich_text list exist
                        parent_content = new_parent_context.setdefault(parent_type, {})
                        current_rich_text_target = parent_content.setdefault("rich_text", [])
                        # Update flag if we are now inside a list item again
                        inside_list_item_direct_content = "list_item" in parent_type if parent_type else False
                     else: # Parent type doesn't have direct rich_text (e.g., column, synced_block)
                         current_rich_text_target = None
                         inside_list_item_direct_content = False
                 else: # Parent context is the root list itself
                     current_rich_text_target = None
                     inside_list_item_direct_content = False
            # Reset target if no pop occurred but needed (e.g., after closing a list).
            elif reset_target:
                 current_rich_text_target = None
                 # Ensure flag is correctly reset if we just closed a regular paragraph/heading.
                 if tag in ["paragraph", "heading"]:
                      inside_list_item_direct_content = False

        # --- 3. Inline Content (inline token) ---
        elif token.type == "inline":
            # This token contains child tokens for text, bold, italic, links, code, math, etc.
            meaningful_children = [c for c in token.children if not ((c.type == 'text' and not c.content.strip()) or c.type == 'softbreak')]
            has_real_content = bool(meaningful_children)
            is_parent_block_implicit = False # Track if we implicitly create a paragraph

            # Handle cases where inline content appears without an active block target
            # (e.g., text directly after a list closes). Create a paragraph implicitly.
            if current_rich_text_target is None and has_real_content and not inside_list_item_direct_content:
                logging.debug("Inline content found without active block context. Creating paragraph.")
                para_block = create_block("paragraph", {"rich_text": []})
                target_list.append(para_block)
                current_rich_text_target = para_block["paragraph"]["rich_text"]
                parent_stack.append(para_block) # Temporarily push this implicit paragraph
                is_parent_block_implicit = True

            # Process inline children if a valid target exists
            if current_rich_text_target is not None:
                # Use snapshots of current formatting state for this sequence of inline elements
                active_annotations_snapshot = active_annotations.copy()
                link_url_snapshot = link_url

                for child in token.children:
                    logging.debug(f"  Inline Child: {child.type} | Content: '{child.content[:30]}...' | Attrs: {getattr(child, 'attrs', '')}")

                    if child.type == "text":
                        cleaned_content = child.content.replace("<br>", "") # Remove explicit breaks
                        if cleaned_content and cleaned_content.strip(): # Append if not just whitespace
                            current_rich_text_target.append(
                                create_rich_text_object(cleaned_content, active_annotations_snapshot, link_url_snapshot)
                            )
                        # Handle spaces: ensure whitespace between elements is preserved if needed
                        elif cleaned_content and not cleaned_content.strip() and current_rich_text_target:
                             last_rt = current_rich_text_target[-1]
                             if last_rt.get("type") == "text" and not last_rt.get("text", {}).get("content", "").endswith(" "):
                                 last_rt["text"]["content"] += " " # Add a space if needed

                    elif child.type == "code_inline":
                         code_annotations = {"code": True, **active_annotations_snapshot}
                         current_rich_text_target.append(create_rich_text_object(child.content, code_annotations, link_url_snapshot))
                    # Toggle annotations based on open/close tags
                    elif child.type == "strong_open": active_annotations_snapshot["bold"] = True
                    elif child.type == "strong_close": active_annotations_snapshot.pop("bold", None)
                    elif child.type == "em_open": active_annotations_snapshot["italic"] = True
                    elif child.type == "em_close": active_annotations_snapshot.pop("italic", None)
                    # Add other styles like strikethrough if the parser supports them
                    elif child.type == "link_open": link_url_snapshot = child.attrs.get('href')
                    elif child.type == "link_close": link_url_snapshot = None
                    elif child.type == "softbreak":
                         # Treat single newlines (softbreaks with breaks=True) as a space
                         if current_rich_text_target and current_rich_text_target[-1].get("type") == "text":
                              if not current_rich_text_target[-1].get("text",{}).get("content","").endswith(" "):
                                  current_rich_text_target[-1]["text"]["content"] += " "
                         logging.debug("Softbreak treated as space.")
                    elif child.type == "hardbreak":
                        # Treat hardbreaks (e.g., double space + newline) as literal '\n'
                        # Notion rendering varies, but this is the closest representation.
                        logging.debug("Hardbreak found. Inserting literal newline '\\n'.")
                        if current_rich_text_target is not None:
                             current_rich_text_target.append(create_rich_text_object("\n", active_annotations_snapshot, link_url_snapshot))
                        else:
                             logging.warning("Hardbreak encountered but no rich text target. Ignoring.")
                    elif child.type == "math_inline":
                         expression = child.content.strip('$ ')
                         if expression:
                             equation_obj = create_equation_object(expression)
                             # Check for placeholder object from empty expression
                             if equation_obj.get("type") == "equation":
                                 current_rich_text_target.append(equation_obj)
                             elif equation_obj.get("type") == "text":
                                  current_rich_text_target.append(equation_obj) # Append placeholder
                         else:
                             logging.warning(f"Skipping empty inline math expression: '{child.content}'")
                    elif child.type == "image":
                         # Images require their own block, even if defined inline in Markdown
                         alt_text = child.content
                         img_url = child.attrs.get('src', '')
                         if img_url:
                             logging.info(f"Creating separate block for image found inline: {img_url}")
                             img_block = create_block("image", {
                                 "type": "external", "external": {"url": img_url},
                                 "caption": [create_rich_text_object(alt_text)] if alt_text else []
                             })
                             # Find the correct parent list to append the image block to
                             # It should be the 'children' list of the block *containing* the current inline token
                             parent_list_for_image = target_list # Usually the correct list
                             parent_list_for_image.append(img_block)
                             # Reset rich text target as image breaks the flow
                             current_rich_text_target = None
                             inside_list_item_direct_content = False
                         else:
                             logging.warning("Image token found inline with no src URL. Skipping.")

            elif not has_real_content:
                 logging.debug("Ignoring inline token with no significant content.")

            # Pop the implicitly created paragraph context if one was added
            if is_parent_block_implicit:
                 parent_stack.pop()

        # --- 4. Standalone Block Elements (Single Token) ---
        elif token.type == "math_block":
            # Reset context, as this is a distinct block type
            current_rich_text_target = None
            inside_list_item_direct_content = False
            expression = token.content.strip('$$\n ') # Remove delimiters and surrounding whitespace
            if expression:
                logging.debug(f"Processing block equation: '{expression[:60]}...'")
                # Create the inner equation object first (handles validation/truncation)
                equation_content_obj = create_equation_object(expression)
                # Check if it returned a valid equation object or a placeholder
                if equation_content_obj.get("type") == "equation":
                     # Create the main block of type "equation" containing the expression
                     new_block = create_block("equation", equation_content_obj["equation"])
                     target_list.append(new_block)
                     logging.debug("Appended block equation.")
                elif equation_content_obj.get("type") == "text": # Was empty, add placeholder paragraph
                     new_block = create_block("paragraph", {"rich_text": [equation_content_obj]})
                     target_list.append(new_block)
                     logging.warning("Added placeholder text block for empty math block.")
            else:
                logging.warning("Math block token found but its expression was empty after stripping.")

        elif token.type == "fence": # Fenced code blocks (``` ... ```)
            current_rich_text_target = None # Code blocks reset the rich text target
            inside_list_item_direct_content = False
            lang_info = token.info.strip().lower() # Language hint (e.g., python, js)
            lang = lang_info or "plain text" # Default to plain text if no hint
            # Map common aliases to Notion's supported language values
            lang_map = {
                'python': 'python', 'py': 'python', 'javascript': 'javascript', 'js': 'javascript',
                'typescript': 'typescript', 'ts': 'typescript', 'html': 'html', 'css': 'css',
                'bash': 'bash', 'shell': 'shell', 'zsh': 'shell', 'json': 'json', 'yaml': 'yaml',
                'markdown': 'markdown', 'sql': 'sql', 'java': 'java', 'kotlin': 'kotlin',
                'c': 'c', 'c++': 'c++', 'cpp': 'c++', 'c#': 'c#', 'csharp': 'c#',
                'ruby': 'ruby', 'rb': 'ruby', 'go': 'go', 'rust': 'rust', 'php': 'php',
                'swift': 'swift', 'plain text': 'plain text', 'text': 'plain text', '': 'plain text'
            }
            notion_lang = lang_map.get(lang, "plain text") # Use mapped value or default
            if lang not in lang_map and lang != "plain text":
                logging.warning(f"Unsupported language hint '{lang}' for code block. Using 'plain text'.")

            code_content = token.content
            # The actual code goes inside a rich_text array within the code block content
            code_rich_text = [create_rich_text_object(code_content)] if code_content else []
            new_block = create_block("code", {
                "rich_text": code_rich_text,
                "language": notion_lang
            })
            target_list.append(new_block)

        elif token.type == "hr": # Horizontal Rule (---, ***, ___)
            current_rich_text_target = None
            inside_list_item_direct_content = False
            # Maps to a Notion divider block, content is always empty.
            new_block = create_block("divider", {})
            target_list.append(new_block)

    # --- Final Debug Output ---
    # if logging.getLogger().isEnabledFor(logging.DEBUG):
    #     logging.debug("--- Final Generated Notion Block Structure ---")
    #     try:
    #         pprint(root_blocks, depth=10, width=120) # Use pretty-print for readability
    #     except Exception as pp_err:
    #         logging.error(f"Error during pretty-printing of final block structure: {pp_err}")
    #         logging.debug(f"Raw final structure: {root_blocks}") # Log raw as fallback
    #     logging.debug("--- End of Generated Block Structure ---")

    return root_blocks

# --- Notion Upload Logic ---

def append_block_tree(parent_block_id: str, blocks: List[Dict[str, Any]], depth=0):
    """
    Recursively appends a list of blocks (and their children) to a Notion parent block.

    Handles:
    - Batching requests to the Notion API (max 100 children per call).
    - Filtering out empty or invalid blocks before sending.
    - Retrying API calls on specific server errors or rate limits.
    - Providing enhanced, user-friendly logging for common API errors.
    - Correctly mapping API response IDs to recurse on created blocks' children.

    Args:
        parent_block_id: The ID of the Notion block to append children to.
        blocks: A list of Notion block dictionaries (potentially with nested 'children').
        depth: The current recursion depth (used for indented logging).
    """
    # Base case for recursion: If there are no blocks to append for this parent, return.
    if not blocks:
        logging.debug(f"{'  ' * depth}No blocks to append under parent {parent_block_id}.")
        return

    indent = "  " * depth # Indentation for logs based on recursion depth
    logging.info(f"{indent}Append Tree | Target Parent ID: {parent_block_id} | Processing {len(blocks)} block(s)...")

    # Process blocks in batches due to Notion API limit (NOTION_MAX_CHILDREN_PER_APPEND)
    for i in range(0, len(blocks), NOTION_MAX_CHILDREN_PER_APPEND):
        batch_number = i // NOTION_MAX_CHILDREN_PER_APPEND + 1
        batch = blocks[i:i + NOTION_MAX_CHILDREN_PER_APPEND]
        logging.debug(f"{indent}Preparing Batch {batch_number} ({len(batch)} blocks) for parent {parent_block_id}...")

        # Store info needed after the API call for potential recursion:
        # Tuple: (block_prepared_for_api, list_of_original_children, original_block_type_string)
        batch_to_send_info: List[Tuple[Dict[str, Any], Optional[List[Dict]], str]] = []

        # --- Prepare and Filter Batch ---
        # Iterate through the current batch to validate, filter empty/invalid blocks,
        # and separate children for later recursive calls.
        for original_block_data in batch:
            # Use deepcopy to avoid modifying the original block list/dict during preparation
            block_copy = copy.deepcopy(original_block_data)

            # --- Separate Children for Later Recursion ---
            # Children must be appended in separate API calls *after* the parent is created.
            # We pop 'children' from the block's *content* dictionary before sending the parent.
            original_children: Optional[List[Dict]] = None
            block_type = block_copy.get("type")
            if block_type and block_type in block_copy and isinstance(block_copy[block_type], dict):
                 # Use pop with default None in case 'children' key doesn't exist (shouldn't happen with create_block)
                 original_children = block_copy[block_type].pop("children", None)

            # --- Basic Block Validation ---
            if not block_type or block_type not in block_copy:
                logging.warning(f"{indent}Skipping invalid block structure in Batch {batch_number}: Missing 'type' or content key. Data: {block_copy}")
                continue # Skip this block

            content_dict = block_copy[block_type]
            is_empty = False # Flag to mark block for filtering

            # --- Filtering Logic (Prevent sending empty/invalid blocks) ---

            # Rule 1: Filter blocks with effectively empty rich_text content,
            # unless the block type is explicitly allowed to be empty (e.g., empty list item).
            if isinstance(content_dict, dict) and 'rich_text' in content_dict:
                rich_text_content = content_dict.get('rich_text', [])
                # Check if rich_text contains any objects with actual text or equations
                is_effectively_empty = not any(
                    (rt.get("type") == "text" and rt.get("text", {}).get("content", "").strip('\n ')) or \
                    rt.get("type") == "equation"
                    for rt in rich_text_content
                )
                # Define block types that are valid even when their rich_text is empty
                allowed_empty_types = {"bulleted_list_item", "numbered_list_item", "toggle", "quote", "callout", "paragraph"}

                if is_effectively_empty and block_type not in allowed_empty_types:
                    logging.debug(f"{indent}Filtering block type '{block_type}' in Batch {batch_number}: Effectively empty rich_text content.")
                    is_empty = True
                elif is_effectively_empty:
                    # Keep allowed empty types, but ensure rich_text list is truly empty
                    content_dict['rich_text'] = []
                else:
                     # Further clean the rich_text list: remove text objects containing only whitespace
                     # (unless it's a deliberate newline character from a hardbreak)
                     cleaned_rich_text = [
                         rt for rt in rich_text_content if
                         (rt.get("type") == "text" and (rt.get("text",{}).get("content", "").strip() or rt.get("text",{}).get("content") == "\n")) or \
                         rt.get("type") == "equation" # Keep all equation objects at this stage
                     ]
                     content_dict['rich_text'] = cleaned_rich_text
                     # Re-check if cleaning made it effectively empty
                     if not cleaned_rich_text and block_type not in allowed_empty_types:
                          logging.debug(f"{indent}Filtering '{block_type}' in Batch {batch_number}: Block became empty after cleaning rich_text.")
                          is_empty = True

            # Rule 2: Filter specific block types if they lack essential data
            elif block_type == "image" and not content_dict.get("external", {}).get("url") and not content_dict.get("file", {}).get("url"):
                 logging.warning(f"{indent}Filtering 'image' block in Batch {batch_number}: Missing required image URL.")
                 is_empty = True
            elif block_type == "equation" and not content_dict.get("expression", "").strip():
                 # create_equation_object should handle this, but safeguard here
                 logging.warning(f"{indent}Filtering 'equation' block in Batch {batch_number}: Contains empty expression.")
                 is_empty = True
            elif block_type == "divider" and not content_dict: # Divider content is {}, which is falsy
                 pass # Valid empty block, do not filter

            # --- Add to Batch if Valid ---
            if is_empty:
                 logging.debug(f"{indent}Skipping filtered empty/invalid block: Type='{block_type}'")
                 continue # Skip this block

            # Store the prepared block, its original children, and type for recursion/debugging
            batch_to_send_info.append((block_copy, original_children, block_type))
        # --- End Batch Preparation ---

        # If the entire batch was filtered out, no need for an API call
        if not batch_to_send_info:
            logging.info(f"{indent}Batch {batch_number} is empty after filtering. Skipping API call.")
            continue

        # Extract the final list of block dictionaries to send in this API request
        prepared_batch_for_api = [info[0] for info in batch_to_send_info]

        # --- API Call with Retries and Enhanced Error Logging ---
        max_retries = 3 # Number of retries for specific errors (5xx, rate limit)
        retry_delay = 1 # Initial delay in seconds
        results = [] # Stores the API response list of created blocks

        for attempt in range(max_retries):
            try:
                logging.debug(f"{indent}Attempt {attempt+1}/{max_retries}: Appending batch {batch_number} ({len(prepared_batch_for_api)} blocks) to parent {parent_block_id}...")
                # *** The Core Notion API Call ***
                response = notion.blocks.children.append(block_id=parent_block_id, children=prepared_batch_for_api)
                results = response.get("results", []) # Extract the list of created blocks from the response

                logging.info(f"{indent}API Success | Batch {batch_number} | Parent: {parent_block_id} | Sent: {len(prepared_batch_for_api)} | Received: {len(results)} results.")

                # --- Sanity Check: Response Count vs Sent Count ---
                # Verify Notion returned a result for each block sent in the batch.
                if len(results) != len(prepared_batch_for_api):
                    logging.error(f"{indent}FATAL API MISMATCH | Batch {batch_number} | Sent {len(prepared_batch_for_api)} blocks but received {len(results)} results.")
                    logging.error(f"{indent}This indicates a potential partial failure or API inconsistency. Aborting this branch.")
                    results = [] # Treat as failure to prevent further processing on incomplete data
                    break # Exit retry loop

                # If successful and counts match, break the retry loop for this batch.
                break

            # --- Handle Notion API Errors (APIResponseError) ---
            except APIResponseError as e:
                # Extract error details safely using getattr for resilience
                error_code = getattr(e, 'code', 'UNKNOWN_CODE')
                error_status = getattr(e, 'status', 0) # HTTP status code
                # str(e) usually contains the helpful message from the Notion API
                error_message_from_api = str(e)
                suggestion = "Check Notion API documentation (developers.notion.com) or Notion status page."
                log_level = logging.ERROR # Default severity for API errors
                should_retry = False # Most client-side errors (4xx) should not be retried

                # --- Map common, user-fixable errors to specific suggestions ---
                if error_code == 'unauthorized' or error_status == 401:
                    log_level = logging.CRITICAL # This is a fundamental configuration issue
                    suggestion = ("Authentication failed. Check if NOTION_API_KEY is correct, valid, and not expired. "
                                  "Ensure the associated integration exists in your Notion settings.")
                elif error_code == 'restricted_resource' or error_status == 403:
                    log_level = logging.CRITICAL # Fundamental access issue
                    api_key_start = notion.options.get('auth', 'UNKNOWN')[:5] # Get start of key for easier ID
                    suggestion = (f"Permission Denied. The integration (Key starting '{api_key_start}...') likely needs to be shared "
                                  f"with the target page/block (ID: {parent_block_id}). Go to the page in Notion, click 'Share' > 'Invite', "
                                  f"find your integration, and grant it 'Can edit' access. Also ensure the page isn't locked.")
                elif error_code == 'object_not_found' or error_status == 404:
                    log_level = logging.CRITICAL # Cannot proceed if target doesn't exist
                    suggestion = (f"The target Notion page or block ID '{parent_block_id}' was not found. "
                                  f"Verify the ID is correct, the page/block exists, and the integration has access.")
                elif error_code == 'validation_error' or error_status == 400:
                    # Provide specific advice based on common validation error messages
                    if "archived" in error_message_from_api.lower():
                         log_level = logging.CRITICAL # Cannot edit archived content
                         suggestion = (f"The target Notion page/block (ID: {parent_block_id}) is archived. "
                                       f"Please restore it from Notion's trash/archive first.")
                    elif "UUID" in error_message_from_api:
                         log_level = logging.CRITICAL # Invalid ID format
                         suggestion = (f"The provided Notion ID '{parent_block_id}' appears invalid or malformed. "
                                       f"Check if it's a correctly formatted UUID (e.g., xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx).")
                    else:
                        # Generic validation error often means invalid block structure/content
                        suggestion = ("Notion reported a 'validation_error'. This often means the data sent for a block was invalid "
                                      "(e.g., wrong content type, missing required field, exceeds limits not caught by script). "
                                      "Check the 'Failed Batch Content' log below (requires --debug flag) for the problematic block.")
                        # Keep log_level as ERROR for generic validation

                # --- Handle Retriable Errors ---
                elif error_code == 'rate_limited' or error_status == 429:
                    log_level = logging.WARNING # Rate limits are recoverable
                    suggestion = "Notion API rate limit hit. Retrying after a delay..."
                    should_retry = True # Flag this attempt for retry
                elif error_status >= 500: # Notion server-side errors (500, 502, 503, 504)
                     log_level = logging.WARNING # Usually temporary
                     suggestion = f"Notion reported a server error (Status: {error_status}). Retrying..."
                     should_retry = True # Flag this attempt for retry

                # --- Log Error Details and Suggestion ---
                logging.log(log_level, f"{indent}NOTION API ERROR | Batch {batch_number} | Attempt {attempt+1}/{max_retries} | Parent: {parent_block_id}")
                logging.log(log_level, f"{indent}Details: Status={error_status}, Code='{error_code}', API Response='{error_message_from_api}'")
                logging.log(log_level, f"{indent}Suggestion: {suggestion}")

                # Log the content of the first block in the failing batch if debug mode is enabled
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    failed_block_preview = prepared_batch_for_api[0] if prepared_batch_for_api else "N/A (Batch was empty?)"
                    logging.debug(f"{indent}--- Failed Batch Content (First Block) ---")
                    try:
                        # Use pprint for better readability, especially for nested structures
                        logging.debug(pprint.pformat(failed_block_preview, indent=2, width=100))
                    except Exception as pp_err:
                         logging.debug(f"{indent}Failed to pretty-print failed block content: {pp_err}. Raw data: {failed_block_preview}")
                    logging.debug(f"{indent}--- End Failed Batch Content ---")

                # --- Retry Logic ---
                # Perform retry only if flagged and attempts remain
                if should_retry and attempt < max_retries - 1:
                    logging.warning(f"{indent}Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Basic exponential backoff (wait longer each time)
                else:
                    # If it shouldn't retry (e.g., 4xx error) or max retries are exhausted
                    if should_retry: # Only log max retries if it was a retriable error
                         logging.error(f"{indent}Max retries reached for batch {batch_number} after server/rate limit errors. Aborting this branch.")
                    else: # Log immediately for non-retriable client errors
                         logging.error(f"{indent}Unrecoverable API error encountered for batch {batch_number}. Aborting this branch.")
                    results = [] # Ensure results is empty to signal failure
                    break # Exit the retry loop for this batch

            # --- Catch Other Unexpected Script Errors ---
            except Exception as e:
                # Catch potential Python errors during the API call/response handling phase
                logging.critical(f"{indent}UNEXPECTED SCRIPT ERROR during Notion API interaction | Batch {batch_number} | Attempt {attempt+1}/{max_retries} | Parent: {parent_block_id}", exc_info=True) # Log full traceback
                logging.critical(f"{indent}Error details: {e}")
                # Optional: Retry even for unexpected errors? Could help with transient network issues.
                if attempt < max_retries - 1:
                     logging.warning(f"{indent}Retrying unexpected error in {retry_delay}s...")
                     time.sleep(retry_delay)
                     retry_delay *= 2
                else:
                     logging.error(f"{indent}Max retries reached for unexpected error. Aborting this branch.")
                     results = [] # Ensure results is empty
                     break # Exit retry loop
        # --- End API Call & Retry Loop for the current batch ---

        # --- Process Results and Initiate Recursion ---
        # Only proceed if the API call for the batch was successful (results list is populated)
        if results:
            # Iterate through the successfully created blocks from the API response
            # and the corresponding original data (including children) from batch_to_send_info.
            for idx, result_block in enumerate(results):
                 # Safety check: Ensure the result index is valid for the batch info we stored.
                 if idx >= len(batch_to_send_info):
                     logging.error(f"{indent}Result index {idx} is out of bounds for sent batch info list (size {len(batch_to_send_info)}). Cannot process children for this result.")
                     continue

                 # Get the ID of the block just created by Notion
                 newly_created_id = result_block.get("id")
                 # Retrieve the corresponding original children list and block type we stored earlier
                 (_, original_children, original_block_type) = batch_to_send_info[idx]

                 # If the block was created successfully (has an ID) AND it originally had children...
                 if newly_created_id and original_children:
                     logging.debug(f"{indent}  Recursion -> Appending {len(original_children)} children under new block ID: {newly_created_id} (Type: {original_block_type})")
                     # Add a small delay before the recursive call, can sometimes help avoid rate limits
                     time.sleep(0.1)
                     # *** Recursive Call *** : Append the children under the newly created parent block
                     append_block_tree(newly_created_id, original_children, depth + 1)
                 elif newly_created_id:
                      # Block created successfully, but it had no children to recurse on.
                      logging.debug(f"{indent}  Leaf Node | Created Block ID: {newly_created_id} (Type: {original_block_type})")
                 else:
                      # This shouldn't happen if the API call succeeded and the sanity check passed, but log just in case.
                      logging.warning(f"{indent}  Skipping recursion: No ID found in successful API result at index {idx}. Original type: {original_block_type}")
        elif len(batch_to_send_info) > 0:
             # If batch_to_send_info was not empty but results is empty, it means the batch failed after retries.
             # We log this failure and implicitly stop processing further batches *within this specific recursive call*,
             # as subsequent blocks might depend on the failed ones for hierarchy.
             logging.error(f"{indent}Batch {batch_number} failed to append to parent {parent_block_id}. Skipping any subsequent batches for this parent.")
             break # Exit the batch processing loop for the current parent_block_id


# --- Main Execution Logic ---
def main():
    """
    Orchestrates the script execution: parses arguments, sets logging level,
    validates inputs, loads Markdown, calls conversion, and initiates the upload.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Convert Markdown file or text to Notion blocks and append to a page/block.",
        formatter_class=argparse.RawTextHelpFormatter # Preserve newlines in help text
    )
    # Define mutually exclusive group for input: user must supply either -f or -t
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--file", metavar="MARKDOWN_FILE", help="Path to the input Markdown file.")
    input_group.add_argument("-t", "--text", metavar='"MARKDOWN_STRING"', help="Markdown text provided directly as a string.\n(Remember to quote strings containing spaces!)")

    # Argument for the target Notion page/block ID
    parser.add_argument(
        "-p", "--page_id",
        default=DEFAULT_NOTION_PAGE_ID, # Use value from .env as default if set
        help="The ID of the Notion page or block where content will be appended.\nDefaults to NOTION_PAGE_ID from .env file if set, otherwise this argument is required."
    )
    # Flag to enable debug logging
    parser.add_argument("--debug", action="store_true", help="Enable detailed DEBUG level logging to console and file.")

    args = parser.parse_args()

    # --- Setup Logging Level (based on --debug argument) ---
    # Get the root logger and console handler configured during initial setup
    root_logger = logging.getLogger()
    global console_handler # Access the globally defined console handler

    is_debug = args.debug
    console_log_level = logging.DEBUG if is_debug else logging.INFO

    # Root logger level is already DEBUG, set console handler level appropriately
    console_handler.setLevel(console_log_level)

    # Adjust Notion client library's own logging level based on debug flag
    global notion # Access the global notion client instance
    # --- MODIFICATION: Commented out dynamic level setting based on --debug to prevent notion_client DEBUG logs ---
    # if 'notion' in globals(): # Check client was initialized successfully
    #     notion.logger.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    # Note: The level set during client initialization (now fixed to WARNING) will persist.

    # Log confirmation of logging levels
    if is_debug:
        # Modified log message to reflect that notion client logging is now fixed
        logging.debug("Debug logging enabled. Console level: DEBUG, File level: DEBUG (File: '%s'). Notion client logging: Fixed at WARNING.", LOG_FILENAME)
    else:
        logging.info("Standard logging enabled. Console level: INFO, File level: DEBUG (File: '%s'). Notion client logging: Fixed at WARNING.", LOG_FILENAME)

    # --- Validate Notion Page/Block ID ---
    target_page_id = args.page_id
    # Critical check: Ensure a target page ID is available (from args or .env)
    if not target_page_id:
        logging.critical("CRITICAL ERROR: Notion Page/Block ID is required but was not provided via -p/--page_id or NOTION_PAGE_ID in .env.")
        sys.exit(1)

    # Validate the format of the provided ID (should be a UUID)
    uuid_pattern_strict = re.compile(r'^[a-f0-9]{8}-?[a-f0-9]{4}-?[1-5][a-f0-9]{3}-?[89ab][a-f0-9]{3}-?[a-f0-9]{12}$') # Includes version check
    uuid_pattern_no_dash = re.compile(r'^[a-f0-9]{32}$') # Plain hex string
    target_page_id_no_dash = target_page_id.replace("-", "") # Normalize for length check

    if not uuid_pattern_strict.match(target_page_id) and not uuid_pattern_no_dash.match(target_page_id_no_dash):
         logging.critical(f"CRITICAL ERROR: Invalid Notion Page/Block ID format provided: '{target_page_id}'.")
         logging.critical("ID should be a valid UUID (e.g., xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx or 32 hexadecimal characters).")
         sys.exit(1)
    elif len(target_page_id_no_dash) != 32: # Double-check length after removing dashes
        logging.critical(f"CRITICAL ERROR: Invalid Notion Page/Block ID length after removing dashes: '{target_page_id}' became '{target_page_id_no_dash}'.")
        logging.critical("Notion IDs must consist of exactly 32 hexadecimal characters.")
        sys.exit(1)

    logging.info(f"Target Notion Page/Block ID validated: {target_page_id}")

    # --- Load Markdown Content ---
    markdown_content: Optional[str] = None
    source_description = "" # For logging where the content came from
    if args.file:
        source_description = f"file '{args.file}'"
        try:
            # Read content from the specified file
            with open(args.file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            logging.info(f"Successfully read Markdown content from {source_description}.")
        except FileNotFoundError:
            logging.critical(f"CRITICAL ERROR: Input Markdown file not found at the specified path: '{args.file}'")
            sys.exit(1)
        except IOError as e:
            logging.critical(f"CRITICAL ERROR: Could not read {source_description}. Check file permissions and path. Error: {e}")
            sys.exit(1)
        except Exception as e: # Catch other potential file errors
            logging.critical(f"CRITICAL ERROR: An unexpected error occurred while reading {source_description}.", exc_info=is_debug)
            sys.exit(1)
    elif args.text:
        # Use the text provided directly as a command-line argument
        source_description = "direct text argument"
        markdown_content = args.text
        logging.info(f"Using Markdown content provided via {source_description}.")
        # Check if the provided text is effectively empty
        if not markdown_content.strip():
             logging.warning("Input markdown text via argument is empty or contains only whitespace.")
             markdown_content = "" # Normalize to empty string

    # Final check: Ensure markdown_content is assigned (should be handled by argparse group)
    if markdown_content is None:
         logging.critical("CRITICAL ERROR: Markdown content could not be loaded from the specified source.")
         sys.exit(1)

    # --- Conversion ---
    logging.info(f"Converting Markdown from {source_description} to Notion block structure...")
    root_blocks = markdown_to_notion_blocks(markdown_content)

    # --- Upload ---
    # Proceed only if the conversion resulted in blocks
    if root_blocks:
        logging.info(f"Conversion generated {len(root_blocks)} top-level block structure(s). Starting upload process to Notion...")
        # Wrap the top-level upload call in try/except for resilience
        try:
            append_block_tree(target_page_id, root_blocks)
            # If append_block_tree finishes without raising an exception that wasn't handled internally.
            logging.info("Upload process finished.")
            logging.info("Please check your Notion page (%s) and review logs (console/file '%s') for details or any warnings/errors during upload.",
                         f"https://www.notion.so/{target_page_id.replace('-', '')}", LOG_FILENAME)
        except Exception as e:
             # Catch errors potentially occurring outside the main API loop in append_block_tree
            logging.critical(f"CRITICAL ERROR: The overall upload process failed with an unexpected error.", exc_info=is_debug)
            logging.critical(f"Error details: {e}")
            sys.exit(1) # Indicate failure
    # Handle cases where conversion yielded no blocks
    elif markdown_content.strip(): # Only warn if the input markdown actually had content
        logging.warning("Markdown conversion resulted in zero Notion blocks. Nothing was uploaded.")
    else: # If the input markdown was empty/whitespace
         logging.info("Input Markdown was empty. Nothing to convert or upload.")

    logging.info("Script finished execution.")


# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        # Execute the main function
        main()
    except SystemExit as e:
         # Catch explicit exits (e.g., from argument parsing errors, failed pre-flight checks)
         # The specific error message should have already been logged by the code calling sys.exit().
         if e.code != 0: # Check if it was an error exit code
              logging.info(f"Script intentionally exited with code {e.code} (indicating an error). Check logs above for the reason.")
         else:
              logging.debug("Script exited cleanly with code 0.") # Typically from --help or successful completion
    except KeyboardInterrupt:
        # Handle user interruption (Ctrl+C) gracefully
        logging.warning("Script execution interrupted by user (KeyboardInterrupt).")
        sys.exit(130) # Standard exit code for SIGINT
    except Exception as e:
         # Catch any other unexpected exceptions at the top level
         logging.critical("An unexpected critical error occurred at the global level.", exc_info=True) # Log full traceback
         sys.exit(1) # Ensure non-zero exit code on unexpected crash
    finally:
        # This block always executes, ensuring logs are finalized.
        logging.info("Script execution ended. Log file located at: %s", os.path.abspath(LOG_FILENAME))
        # Explicitly shutdown logging to flush/close handlers, especially the file handler.
        logging.shutdown()
